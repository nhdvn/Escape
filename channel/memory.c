#include "channel.h"
#include <stdlib.h>
#include <string.h>

static void memory_send_sub_rows(channel_t *channel, const int *sub_rows)
{
    memcpy(channel->sub_rows, sub_rows, PLHE_N1 * sizeof(int));
}

static void memory_recv_sub_rows(channel_t *channel, int *sub_rows)
{
    memcpy(sub_rows, channel->sub_rows, PLHE_N1 * sizeof(int));
}

static void memory_send_enc_bits(channel_t *channel, const plhe_q_t *enc_bits)
{
    memcpy(channel->enc_bits, enc_bits, (size_t)PLHE_N1 * PLHE_MC1 * sizeof(plhe_q_t));
}

static void memory_recv_enc_bits(channel_t *channel, plhe_q_t *enc_bits)
{
    memcpy(enc_bits, channel->enc_bits, (size_t)PLHE_N1 * PLHE_MC1 * sizeof(plhe_q_t));
}

/* Local channel: ctx is shared by pointer (no marshalling).                 */
static void memory_send_context(channel_t *channel, compress_ctx_t *ctx)
{
    channel->context = ctx;
}
static compress_ctx_t *memory_recv_context(channel_t *channel)
{
    return channel->context;
}

/* Local channel: answer output passes through a channel-owned mpz_t
 * array. Caller-side buffers stay separate (proper send/recv copy).         */
static void ensure_answer_buf(channel_t *channel, int n_out)
{
    if (channel->answer_n == n_out) return;
    if (channel->answer) {
        for (int i = 0; i < channel->answer_n; i++)
            mpz_clear(channel->answer[i]);
        free(channel->answer);
    }
    channel->answer = malloc((size_t)n_out * sizeof(mpz_t));
    for (int i = 0; i < n_out; i++) mpz_init(channel->answer[i]);
    channel->answer_n = n_out;
}

static void memory_send_answer(channel_t *channel, const mpz_t *out, int n_out)
{
    ensure_answer_buf(channel, n_out);
    for (int i = 0; i < n_out; i++) mpz_set(channel->answer[i], out[i]);
}

static void memory_recv_answer(channel_t *channel, mpz_t *out, int n_out)
{
    for (int i = 0; i < n_out; i++) mpz_set(out[i], channel->answer[i]);
}

channel_t *memory_create(void)
{
    channel_t *channel = calloc(1, sizeof(channel_t));
    channel->send_sub_rows = memory_send_sub_rows;
    channel->recv_sub_rows = memory_recv_sub_rows;
    channel->send_enc_bits = memory_send_enc_bits;
    channel->recv_enc_bits = memory_recv_enc_bits;
    channel->send_context  = memory_send_context;
    channel->recv_context  = memory_recv_context;
    channel->send_answer   = memory_send_answer;
    channel->recv_answer   = memory_recv_answer;
    return channel;
}

void channel_free(channel_t *channel)
{
    if (!channel) return;
    if (channel->answer) {
        for (int i = 0; i < channel->answer_n; i++)
            mpz_clear(channel->answer[i]);
        free(channel->answer);
    }
    free(channel);
}
