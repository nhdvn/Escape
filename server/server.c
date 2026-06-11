#include "server.h"
#include "../plhe/plhe.h"
#include "../compress/compress.h"
#include "../utils/db.h"
#include "../utils/measure.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>

static inline void bench_db_row(int k, int row, uint8_t *out)
{
#if BENCH_SKIP_GEN
    (void)k; (void)row; (void)out;
#else
    virtual_db_row(k, row, out);
#endif
}

static inline void bench_r_hint(plhe_q_t *r_hint, const uint8_t *db_row, int p_row)
{
#if BENCH_SKIP_GEN
    (void)r_hint; (void)db_row; (void)p_row;
#else
    plhe_dhint_row(r_hint, db_row, p_row);
#endif
}

typedef struct {
    const int       *sub_rows;
    const plhe_q_t (*enc_bits)[PLHE_MC1];
    plhe_q_t        *ans;       /* [B_CHUNKS]            partial ans */
    plhe_q_t        *agg;       /* [B_CHUNKS * LAMBDA]   partial agg */
    uint64_t         delay_cycle, delay_stall, delay_memory;
    long             delay_total;
    int              k_start, k_finis;
} worker_t;

static void *worker_fn(void *arg)
{
    worker_t *wr = arg;
    plhe_q_t *r_hints[AGG_BATCH];
    uint8_t  *db_row;
    for (int b = 0; b < AGG_BATCH; b++)
        ALLOC_ALIGN64(r_hints[b], B_CHUNKS * PLHE_LAMBDA * sizeof(plhe_q_t));
    ALLOC_ALIGN64(db_row, PLHE_MC1 * B_CHUNKS);

#if BENCH_SKIP_GEN
    random_buffer(db_row, (size_t)PLHE_MC1 * B_CHUNKS);
    for (int b = 0; b < AGG_BATCH; b++)
        random_buffer(r_hints[b], (size_t)B_CHUNKS * PLHE_LAMBDA * sizeof(plhe_q_t));
#endif

    /* Hoist SIMD-inner pointers: TBAA assumes *plhe_q_t aliases worker_t. */
    plhe_q_t *ans = wr->ans;
    plhe_q_t *agg = wr->agg;
    const plhe_q_t (*enc_bits)[PLHE_MC1] = wr->enc_bits;

    long delay_total = 0;
    uint64_t delay_cycle = 0, delay_stall = 0, delay_memory = 0;
    measure_t pc = MEASURE_INIT;
    int pc_ok = (measure_init(&pc) == 0);
    int k = wr->k_start;

    for (; k + AGG_BATCH <= wr->k_finis; k += AGG_BATCH) {
        for (int b = 0; b < AGG_BATCH; b++) {
            int kk    = k + b;
            int p_row = kk * PLHE_MR1 + wr->sub_rows[kk];
            bench_db_row(kk, wr->sub_rows[kk], db_row);   /* enable this for correctness will thrash cpu */
            bench_r_hint(r_hints[b], db_row, p_row);    /* this is because we are simulating large db */
            MEASURE_START();
            for (int col = 0; col < PLHE_MC1; col++) {
                const uint8_t *db_col = db_row + col * B_CHUNKS;
                U8_FMA_LOOP(ans, db_col, enc_bits[kk][col], B_CHUNKS);
            }
            MEASURE_FINIS();
        }
        MEASURE_START();
        U32_ADDN_LOOP_ALIGNED(agg, r_hints, AGG_BATCH, B_CHUNKS * PLHE_LAMBDA);
        MEASURE_FINIS();
    }

    /* Tail: < AGG_BATCH remaining ks, single-k path */
    for (; k < wr->k_finis; k++) {
        int p_row = k * PLHE_MR1 + wr->sub_rows[k];
        bench_db_row(k, wr->sub_rows[k], db_row);
        bench_r_hint(r_hints[0], db_row, p_row);
        MEASURE_START();
        for (int col = 0; col < PLHE_MC1; col++) {
            const uint8_t *db_col = db_row + col * B_CHUNKS;
            U8_FMA_LOOP(ans, db_col, enc_bits[k][col], B_CHUNKS);
        }
        U32_ADD_LOOP_ALIGNED(agg, r_hints[0], B_CHUNKS * PLHE_LAMBDA);
        MEASURE_FINIS();
    }

    if (pc_ok) measure_close(&pc);

    wr->delay_total  = delay_total;
    wr->delay_cycle  = delay_cycle;
    wr->delay_stall  = delay_stall;
    wr->delay_memory = delay_memory;

    free(db_row);
    for (int b = 0; b < AGG_BATCH; b++) free(r_hints[b]);
    return NULL;
}

/* -- server lifecycle ---------------------------------------------------- */

struct server_s {
    int _dummy;
};

server_t *server_create(void)
{
    return calloc(1, sizeof(server_t));
}

void server_destroy(server_t *server)
{
    free(server);
}

/* -- online phase -------------------------------------------------------- */

void server_answer(server_t *server, channel_t *channel)
{
    /* Build zkw up front so it overlaps with the enc_bits upload time. */
    compress_ctx_t *cntx = channel->recv_context(channel);
    omp_set_num_threads(SERVER_THREADS);
    precompute_window(cntx);

    int sub_rows[PLHE_N1];
    plhe_q_t *enc_bits = malloc((size_t)PLHE_N1 * PLHE_MC1 * sizeof(plhe_q_t));
    channel->recv_sub_rows(channel, sub_rows);
    channel->recv_enc_bits(channel, enc_bits);

    (void)server;
    worker_t  workers[SERVER_THREADS];
    pthread_t pids   [SERVER_THREADS];

    for (int t = 0; t < SERVER_THREADS; t++) {
        worker_t *w = &workers[t];
        w->sub_rows = sub_rows;
        w->enc_bits = (const plhe_q_t (*)[PLHE_MC1])enc_bits;
        w->k_start  = t * PLHE_N1 / SERVER_THREADS;
        w->k_finis  = (t + 1) * PLHE_N1 / SERVER_THREADS;
        ALLOC_ALIGN64(w->agg, B_CHUNKS * PLHE_LAMBDA * sizeof(plhe_q_t));
        ALLOC_ALIGN64(w->ans, B_CHUNKS * sizeof(plhe_q_t));
        pthread_create(&pids[t], NULL, worker_fn, w);
    }
    for (int t = 0; t < SERVER_THREADS; t++)
        pthread_join(pids[t], NULL);

    free(enc_bits);

    /* Reduce per-worker partials into the final ans and agg. */
    plhe_q_t  ans[B_CHUNKS] __attribute__((aligned(64))) = {0};
    plhe_q_t *agg;
    ALLOC_ALIGN64(agg, B_CHUNKS * PLHE_LAMBDA * sizeof(plhe_q_t));
    const plhe_q_t *ans_srcs[SERVER_THREADS];
    const plhe_q_t *agg_srcs [SERVER_THREADS];
    long max_total = 0;
    uint64_t max_cycle = 0, max_stall = 0, max_memory = 0;

    for (int t = 0; t < SERVER_THREADS; t++) {
        ans_srcs[t] = workers[t].ans;
        agg_srcs[t] = workers[t].agg;
        if (workers[t].delay_total > max_total) {
            max_total  = workers[t].delay_total;
            max_cycle  = workers[t].delay_cycle;
            max_stall  = workers[t].delay_stall;
            max_memory = workers[t].delay_memory;
        }
    }

    U32_ADDN_LOOP_ALIGNED(ans, ans_srcs, SERVER_THREADS, B_CHUNKS);
    U32_ADDN_LOOP_ALIGNED(agg, agg_srcs, SERVER_THREADS, B_CHUNKS * PLHE_LAMBDA);

    if (max_cycle > 0) {
        double freq = (double)max_cycle / ((double)max_total / 1e9);
        double cpu_total  = max_total / 1e6;
        double ans_cycle  = (double)(max_cycle - max_stall) / freq * 1e3;
        double cpu_stall  = (double)(max_stall - max_memory) / freq * 1e3;
        double mem_stall  = (double)(max_memory) / freq * 1e3;
        printf("  [server ans]   %.1f ms\n", cpu_total);
        printf("    ans_cycle    %.1f ms\n", ans_cycle);
        printf("    cpu_stall    %.1f ms\n", cpu_stall);
        printf("    mem_stall    %.1f ms  (%.0f%% of answer)\n",
               mem_stall, 100.0 * mem_stall / cpu_total);
    } else {
        printf("  [server ans]   %.1f ms\n", max_total / 1e6);
    }

    for (int t = 0; t < SERVER_THREADS; t++) {
        free(workers[t].ans);
        free(workers[t].agg);
    }

    /* Compress (agg, ans) into n_out packed Paillier ciphertexts. */
    int n_out = cntx->n_out;
    mpz_t *out = malloc((size_t)n_out * sizeof(mpz_t));
    for (int i = 0; i < n_out; i++) mpz_init(out[i]);

    compress_response(cntx, agg, ans, out);
    channel->send_answer(channel, out, n_out);

    for (int i = 0; i < n_out; i++) mpz_clear(out[i]);
    free(out);
    free(agg);
}
