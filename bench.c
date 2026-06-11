#include "plhe/params.h"
#include "plhe/plhe.h"
#include "utils/db.h"
#include "utils/measure.h"
#include "channel/channel.h"
#include "server/server.h"
#include "client/client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_params(void)
{
    size_t n_db_entries = (size_t)PLHE_N1 * PLHE_MR1 * PLHE_MC1;
    printf("-- Escape v1 Parameters --\n");
    printf("  Q=%d  P=%d  Delta=2^%d  lambda=%d  N1=%d  MR1=%d  MC1=%d  sigma=%.1f\n",
           PLHE_Q_BITS, PLHE_P_BITS, PLHE_Q_BITS - PLHE_P_BITS,
           PLHE_LAMBDA, PLHE_N1, PLHE_MR1, PLHE_MC1,
           (double)PLHE_SIGMA_X10 / 10.0);
    printf("  threads    = (%d, %d)\n", SERVER_THREADS, CLIENT_THREADS);
    printf("  logical DB = %zu entries (%.2f GiB)\n",
           n_db_entries, (double)n_db_entries * BLOCK_SIZE_BITS / 8 / (1u << 30));
    printf("  with dhint = %.2f GiB\n",
           (double)PLHE_N1 * PLHE_MR1 * B_CHUNKS * PLHE_LAMBDA * 4 / (1u << 30));
    printf("  agg        = %.2f MiB\n",
           (double)B_CHUNKS * PLHE_LAMBDA * 8 / (1u << 20));
    printf("  block_size = %d bits  ->  %d chunk(s) per query\n",
           BLOCK_SIZE_BITS, B_CHUNKS);
    printf("--------------------------\n\n");
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    int n_reps = (argc > 1) ? atoi(argv[1]) : 5;
    if (n_reps < 1) n_reps = 1;

    print_params();

    server_t    *server  = server_create();
    channel_t   *channel = memory_create();
    client_t    *client  = client_create();

    int sub_rows[PLHE_N1], col_indx[PLHE_N1];

    /* -- warmup -- */
    plhe_rng_t warmup_rng;
    plhe_rng_init(&warmup_rng, 0xc0ffee1234567890ULL);
    for (int k = 0; k < PLHE_N1; k++) {
        sub_rows[k] = (int)(plhe_rng(&warmup_rng) % PLHE_MR1);
        col_indx[k] = (int)(plhe_rng(&warmup_rng) % PLHE_MC1);
    }
    uint8_t warmup_results[B_CHUNKS];
    for (int r = 0; r < 3; r++) {
        client_query(client, channel, sub_rows, col_indx);
        server_answer(server, channel);
        client_recover(client, channel, warmup_results);
    }

    /* -- bench -- */
    plhe_rng_t bench_rng;
    plhe_rng_init(&bench_rng, 0xabcdef1234567890ULL);
    int n_errors = 0;

    /* heap-allocated to avoid stack overflow at large MC1/B_CHUNKS */
    uint8_t *db_row_phys = malloc((size_t)PLHE_MC1 * B_CHUNKS);

    for (int rep = 0; rep < n_reps; rep++) {
        for (int k = 0; k < PLHE_N1; k++) {
            sub_rows[k] = (int)(plhe_rng(&bench_rng) % PLHE_MR1);
            col_indx[k] = (int)(plhe_rng(&bench_rng) % PLHE_MC1);
        }

        uint8_t results[B_CHUNKS];
        client_query(client, channel, sub_rows, col_indx);
        server_answer(server, channel);
        client_recover(client, channel, results);

        uint8_t expected[B_CHUNKS];
        memset(expected, 0, sizeof(expected));
        for (int k = 0; k < PLHE_N1; k++) {
            virtual_db_row(k, sub_rows[k], db_row_phys);
            for (int chunk = 0; chunk < B_CHUNKS; chunk++)
                expected[chunk] += db_row_phys[col_indx[k] * B_CHUNKS + chunk];
        }
        for (int chunk = 0; chunk < B_CHUNKS; chunk++)
            if (results[chunk] != expected[chunk] && n_errors++ < 5)
                printf("[ERROR] rep=%d chunk=%d: got %u expected %u\n",
                       rep, chunk, results[chunk], expected[chunk]);
    }

    printf("\n\n[online] reps=%d  chunks=%d\n", n_reps, B_CHUNKS);
    if (n_errors) printf("\nFAIL: %d error(s)\n", n_errors);
    else          printf("\nOK\n");

    free(db_row_phys);
    client_destroy(client);
    server_destroy(server);
    channel_free(channel);
    return n_errors ? 1 : 0;
}
