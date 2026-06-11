#pragma once
#include "../plhe/params.h"
#include "../compress/compress.h"
#include <stddef.h>

typedef struct channel_s channel_t;

struct channel_s {
    void (*send_sub_rows)(channel_t *channel, const int *sub_rows);
    void (*recv_sub_rows)(channel_t *channel,       int *sub_rows);

    void (*send_enc_bits)(channel_t *channel, const plhe_q_t *enc_bits);
    void (*recv_enc_bits)(channel_t *channel,       plhe_q_t *enc_bits);

    /* Compression handshake.
     *   send_context: client publishes the compress_ctx_t (carries zkey for
     *                    the server's zkw precompute, plus public Paillier
     *                    params). In the memory channel this is a pointer
     *                    pass-through; for a network channel it would marshal
     *                    only the public side (n, n_sq, zkey).
     *   send_answer: server publishes n_out packed Paillier ciphertexts.
     *                    Caller pre-mpz_init's all entries on both sides.    */
    void (*send_context)(channel_t *channel, compress_ctx_t *ctx);
    compress_ctx_t *(*recv_context)(channel_t *channel);

    void (*send_answer)(channel_t *channel, const mpz_t *out, int n_out);
    void (*recv_answer)(channel_t *channel,       mpz_t *out, int n_out);

    int       sub_rows[PLHE_N1];
    plhe_q_t  enc_bits[PLHE_N1 * PLHE_MC1];
    compress_ctx_t *context;    /* shared pointer (memory channel) */
    mpz_t    *answer;           /* [n_out] */
    int       answer_n;         /* current n_out                 */
};

channel_t *memory_create(void);
void       channel_free(channel_t *channel);
