#include "client.h"
#include "../plhe/plhe.h"
#include "../compress/compress.h"
#include "../utils/barrier.h"
#include "../utils/measure.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

/* -- query worker --------------------------------------------------------- */

typedef struct {
    const uint8_t *sk;                       /* shared across partitions */
    plhe_q_t      (*enc_bits)[PLHE_MC1];
    const int     *sub_rows;
    const int     *col_indx;
    volatile int   alive;
    plhe_rng_t     rng;
    barrier_t     *b_start, *b_finis;
    int            k_start, k_finis;
} worker_t;

static void *worker_fn(void *arg)
{
    worker_t *w = arg;
    while (1) {
        barrier_wait(w->b_start);
        if (!w->alive) break;
        for (int k = w->k_start; k < w->k_finis; k++) {
            int p_row = k * PLHE_MR1 + w->sub_rows[k];
            plhe_encrypt_vbit(w->enc_bits[k], p_row,
                              w->col_indx[k], w->sk, &w->rng);
        }
        barrier_wait(w->b_finis);
    }
    return NULL;
}

/* -- client lifecycle ----------------------------------------------------- */

struct client_s {
    plhe_q_t   enc_bits[PLHE_N1][PLHE_MC1];
    plhe_rng_t rng;
    int        sub_rows[PLHE_N1];
    int        col_indx[PLHE_N1];

    compress_ctx_t  *cntx;  /* sk + zkey + Paillier keys + zkw table */
    barrier_t   b_start, b_finis;
    pthread_t   pids   [CLIENT_THREADS];
    worker_t    workers[CLIENT_THREADS];
};

client_t *client_create(void)
{
    client_t *client = calloc(1, sizeof(client_t));
    plhe_rng_init(&client->rng, now_ns() + 1);

    client->cntx = compress_ctx_setup(PLHE_LAMBDA, B_CHUNKS,
                                       PLHE_Q_BITS, PLHE_P_BITS, PL_BITS);
    precompute_random(client->cntx, CLIENT_THREADS);

    barrier_init(&client->b_start, CLIENT_THREADS + 1);
    barrier_init(&client->b_finis, CLIENT_THREADS + 1);

    int k_per = PLHE_N1 / CLIENT_THREADS;
    for (int t = 0; t < CLIENT_THREADS; t++) {
        worker_t *worker = &client->workers[t];
        worker->sk       = client->cntx->sk;
        worker->enc_bits = client->enc_bits;
        worker->sub_rows = NULL;
        worker->col_indx = NULL;
        worker->alive    = 1;
        worker->b_start  = &client->b_start;
        worker->b_finis  = &client->b_finis;
        worker->k_start  = t * k_per;
        worker->k_finis  = (t == CLIENT_THREADS - 1) ? PLHE_N1 : (t + 1) * k_per;
        plhe_rng_init(&worker->rng, plhe_rng(&client->rng));
        pthread_create(&client->pids[t], NULL, worker_fn, worker);
    }
    return client;
}

void client_destroy(client_t *client)
{
    if (!client) return;
    for (int t = 0; t < CLIENT_THREADS; t++)
        client->workers[t].alive = 0;
    barrier_wait(&client->b_start);
    for (int t = 0; t < CLIENT_THREADS; t++)
        pthread_join(client->pids[t], NULL);
    barrier_destroy(&client->b_start);
    barrier_destroy(&client->b_finis);
    compress_ctx_free(client->cntx);
    free(client);
}

/* -- query / recover ------------------------------------------------------ */

void client_query(client_t *client, channel_t *channel,
                  const int *sub_rows, const int *col_indx)
{
    printf("--------------------------------------------------------\n");
    compress_key(client->cntx, CLIENT_THREADS);

    /* Ship cntx now so its upload overlaps with enc_bits compute below. */
    channel->send_context(channel, client->cntx);

    for (int t = 0; t < CLIENT_THREADS; t++)
        client->workers[t].sk = client->cntx->sk;

    memcpy(client->sub_rows, sub_rows, PLHE_N1 * sizeof(int));
    memcpy(client->col_indx, col_indx, PLHE_N1 * sizeof(int));
    for (int t = 0; t < CLIENT_THREADS; t++) {
        client->workers[t].sub_rows = client->sub_rows;
        client->workers[t].col_indx = client->col_indx;
    }

    uint64_t t0 = now_ns();
    barrier_wait(&client->b_start);
    barrier_wait(&client->b_finis);
    printf("  [query comp]   %.1f ms\n", (now_ns() - t0) / 1e6);

    channel->send_sub_rows(channel, sub_rows);
    channel->send_enc_bits(channel, &client->enc_bits[0][0]);

    /* zkey + rows + bits */
    size_t send_bytes = (size_t)client->cntx->lmbda * (client->cntx->pl_bits * 2 / 8)
        + (size_t)PLHE_N1 * sizeof(int) + (size_t)PLHE_N1 * PLHE_MC1 * sizeof(plhe_q_t);
    
    double send_ms = (double)send_bytes * 8.0 / (NET_MBPS * 1e6) * 1e3;
    printf("  [client send]  %.3f MiB  ->  %.1f ms @ %d Mbps\n",
           (double)send_bytes / (1024.0 * 1024.0), send_ms, NET_MBPS);
}

void client_recover(client_t *client, channel_t *channel, uint8_t result[B_CHUNKS])
{
    int n_out = client->cntx->n_out;

    mpz_t *answer = malloc((size_t)n_out * sizeof(mpz_t));
    for (int i = 0; i < n_out; i++) mpz_init(answer[i]);
    channel->recv_answer(channel, answer, n_out);

    size_t recv_bytes = (size_t)n_out * (client->cntx->pl_bits * 2 / 8);
    double recv_ms = (double)recv_bytes * 8.0 / (NET_MBPS * 1e6) * 1e3;
    printf("  [client recv]  %.3f KiB  ->  %.1f ms @ %d Mbps\n",
           (double)recv_bytes / 1024.0, recv_ms, NET_MBPS);

    uint64_t t0 = now_ns();
    decompress_response(client->cntx, answer, result, CLIENT_THREADS);
    printf("  [recover comp] %.1f ms\n", (now_ns() - t0) / 1e6);

    for (int i = 0; i < n_out; i++) mpz_clear(answer[i]);
    free(answer);
}
