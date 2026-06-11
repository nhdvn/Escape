/* _gmpz.c -- portable GMP fallback for the compress.h API.  Selected by
 * Makefile when AVX-512 IFMA52 is unavailable.  All Paillier arithmetic via
 * mpz_t.  Shared helpers (mod_switch_u32, paillier_keygen, ceil_log2_u32,
 * gen_safe_prime) live in compress.h.                                       */
#include "compress.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void paillier_decrypt(mpz_t out, const compress_ctx_t *ctx, const mpz_t c)
{
    mpz_t t;
    mpz_init(t);
    mpz_powm(t, c, ctx->d_xp, ctx->n_sq);
    mpz_sub_ui(t, t, 1);
    mpz_divexact(t, t, ctx->n);
    mpz_mul(out, t, ctx->mu);
    mpz_mod(out, out, ctx->n);
    mpz_clear(t);
}

/* ---------- Context lifecycle ----------------------------------------- */

compress_ctx_t *compress_ctx_setup(int lmbda, int n_cts,
                                   int log_q, int log_p, int pl_bits)
{
    compress_ctx_t *ctx = calloc(1, sizeof(*ctx));
    ctx->lmbda    = lmbda;
    ctx->n_cts    = n_cts;
    ctx->log_q    = log_q;
    ctx->log_p    = log_p;
    ctx->pl_bits  = pl_bits;

    int headroom  = ceil_log2_u32((uint32_t)(lmbda + 1));
    ctx->log_r    = log_p + headroom + 2;
    ctx->n_slots  = (pl_bits - headroom) / ctx->log_r;
    ctx->n_out    = (n_cts + ctx->n_slots - 1) / ctx->n_slots;

    mpz_inits(ctx->n, ctx->n_sq, ctx->d_xp, ctx->mu, ctx->shift_pow, NULL);

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 0xdeadcafeUL);
    paillier_keygen(ctx, rs);
    gmp_randclear(rs);

    mpz_set_ui(ctx->shift_pow, 1);
    mpz_mul_2exp(ctx->shift_pow, ctx->shift_pow, (mp_bitcnt_t)ctx->log_r);

    return ctx;
}

void compress_ctx_free(compress_ctx_t *ctx)
{
    if (!ctx) return;
    mpz_clears(ctx->n, ctx->n_sq, ctx->d_xp, ctx->mu, ctx->shift_pow, NULL);
    if (ctx->zkey) {
        for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->zkey[i]);
        free(ctx->zkey);
    }
    if (ctx->zkw) {
        int n_entries = (1 << ctx->zkw_bits) - 1;
        for (int d = 0; d < ctx->zkn_digits; d++) {
            for (int k = 0; k < n_entries; k++) {
                for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->zkw[d][k][i]);
                free(ctx->zkw[d][k]);
            }
            free(ctx->zkw[d]);
        }
        free(ctx->zkw);
    }
    if (ctx->r_pow_n) {
        for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->r_pow_n[i]);
        free(ctx->r_pow_n);
    }
    free(ctx->sk);
    free(ctx);
}

/* ---------- Pre-computation (per-query) ------------------------------- */

void precompute_random(compress_ctx_t *ctx, int n_thrds)
{
    int lmbda = ctx->lmbda;
    if (n_thrds < 1) n_thrds = 1;

    if (ctx->r_pow_n) {
        for (int i = 0; i < lmbda; i++) mpz_clear(ctx->r_pow_n[i]);
        free(ctx->r_pow_n);
    }
    ctx->r_pow_n = malloc((size_t)lmbda * sizeof(mpz_t));
    for (int i = 0; i < lmbda; i++) mpz_init(ctx->r_pow_n[i]);

    #pragma omp parallel num_threads(n_thrds)
    {
        mpz_t r; mpz_init(r);
        gmp_randstate_t rs; gmp_randinit_default(rs);
        gmp_randseed_ui(rs, 0xdec0de + omp_get_thread_num() * 31);

        #pragma omp for schedule(static)
        for (int i = 0; i < lmbda; i++) {
            mpz_urandomm(r, rs, ctx->n);
            mpz_powm(ctx->r_pow_n[i], r, ctx->n, ctx->n_sq);
        }
        mpz_clear(r);
        gmp_randclear(rs);
    }
}

void compress_key(compress_ctx_t *ctx, int n_thrds)
{
    int lmbda = ctx->lmbda;
    if (n_thrds < 1) n_thrds = 1;

    if (ctx->zkey) {
        for (int i = 0; i < lmbda; i++) mpz_clear(ctx->zkey[i]);
        free(ctx->zkey);
    }
    if (ctx->sk) free(ctx->sk);

    ctx->sk   = malloc((size_t)lmbda);
    ctx->zkey = malloc((size_t)lmbda * sizeof(mpz_t));

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 0xfeedfaceUL);

    for (int i = 0; i < lmbda; i++) {
        ctx->sk[i] = (uint8_t)(gmp_urandomb_ui(rs, 1));
        mpz_init(ctx->zkey[i]);
    }
    gmp_randclear(rs);

    #pragma omp parallel num_threads(n_thrds)
    {
        mpz_t bn;
        mpz_init(bn);
        #pragma omp for schedule(static)
        for (int i = 0; i < lmbda; i++) {
            mpz_set_ui(bn, ctx->sk[i]);
            mpz_mul(bn, bn, ctx->n);
            mpz_add_ui(bn, bn, 1);
            mpz_mul(ctx->zkey[i], bn, ctx->r_pow_n[i]);
            mpz_mod(ctx->zkey[i], ctx->zkey[i], ctx->n_sq);
        }
        mpz_clear(bn);
    }
}

void precompute_window(compress_ctx_t *ctx)
{
    int lmbda = ctx->lmbda;

    if (ctx->zkw) {
        int old_entries = (1 << ctx->zkw_bits) - 1;
        for (int d = 0; d < ctx->zkn_digits; d++) {
            for (int k = 0; k < old_entries; k++) {
                for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->zkw[d][k][i]);
                free(ctx->zkw[d][k]);
            }
            free(ctx->zkw[d]);
        }
        free(ctx->zkw);
        ctx->zkw = NULL;
    }

    ctx->zkw_bits   = ZKW;
    ctx->zkn_digits = (ctx->log_r + ZKW - 1) / ZKW;
    int n_digits  = ctx->zkn_digits;
    int n_entries = (1 << ZKW) - 1;

    ctx->zkw = malloc((size_t)n_digits * sizeof(mpz_t **));
    for (int d = 0; d < n_digits; d++) {
        ctx->zkw[d] = malloc((size_t)n_entries * sizeof(mpz_t *));
        for (int k = 0; k < n_entries; k++) {
            ctx->zkw[d][k] = malloc((size_t)lmbda * sizeof(mpz_t));
            for (int i = 0; i < lmbda; i++) mpz_init(ctx->zkw[d][k][i]);
        }
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < lmbda; i++) mpz_set(ctx->zkw[0][0][i], ctx->zkey[i]);

        for (int d = 1; d < n_digits; d++) {
            #pragma omp for schedule(static)
            for (int i = 0; i < lmbda; i++) {
                mpz_set(ctx->zkw[d][0][i], ctx->zkw[d-1][0][i]);
                for (int s = 0; s < ZKW; s++) {
                    mpz_mul(ctx->zkw[d][0][i], ctx->zkw[d][0][i], ctx->zkw[d][0][i]);
                    mpz_mod(ctx->zkw[d][0][i], ctx->zkw[d][0][i], ctx->n_sq);
                }
            }
        }
        for (int d = 0; d < n_digits; d++) {
            for (int k = 1; k < n_entries; k++) {
                #pragma omp for schedule(static)
                for (int i = 0; i < lmbda; i++) {
                    mpz_mul(ctx->zkw[d][k][i], ctx->zkw[d][k-1][i], ctx->zkw[d][0][i]);
                    mpz_mod(ctx->zkw[d][k][i], ctx->zkw[d][k][i], ctx->n_sq);
                }
            }
        }
    }
}

/* ---------- Online compress / decompress ------------------------------ */

int compress_response(compress_ctx_t *ctx, const uint32_t *vA,
                      const uint32_t *vb, mpz_t *out)
{
    int      lmbda    = ctx->lmbda;
    int      n_cts    = ctx->n_cts;
    int      n_slots  = ctx->n_slots;
    int      log_r    = ctx->log_r;
    int      n_out    = ctx->n_out;
    int      n_digits = ctx->zkn_digits;
    uint32_t r_mask   = (log_r >= 32) ? 0xffffffffu : ((1u << log_r) - 1u);
    uint32_t w_mask   = (1u << ZKW) - 1u;

    uint32_t *a_prime = malloc((size_t)n_cts * lmbda * sizeof(uint32_t));
    uint32_t *b_prime = malloc((size_t)n_cts * sizeof(uint32_t));
    mod_switch_u32(a_prime, vA, (size_t)n_cts * lmbda, ctx->log_q, log_r);
    mod_switch_u32(b_prime, vb, (size_t)n_cts,         ctx->log_q, log_r);

    mpz_t *x = malloc((size_t)n_cts * sizeof(mpz_t));
    for (int c = 0; c < n_cts; c++) mpz_init(x[c]);

    #pragma omp parallel
    {
        mpz_t bn; mpz_init(bn);

        #pragma omp for schedule(static)
        for (int c = 0; c < n_cts; c++) {
            mpz_set_ui(bn, b_prime[c]);
            mpz_mul(bn, bn, ctx->n);
            mpz_add_ui(bn, bn, 1);
            mpz_mod(x[c], bn, ctx->n_sq);

            const uint32_t *a_row = a_prime + (size_t)c * lmbda;
            for (int i = 0; i < lmbda; i++) {
                uint32_t scal = ((uint32_t)0 - a_row[i]) & r_mask;
                for (int d = 0; d < n_digits && scal; d++, scal >>= ZKW) {
                    uint32_t digit = scal & w_mask;
                    if (digit == 0) continue;
                    mpz_mul(x[c], x[c], ctx->zkw[d][digit - 1][i]);
                    mpz_mod(x[c], x[c], ctx->n_sq);
                }
            }
        }

        #pragma omp for schedule(static)
        for (int out_idx = 0; out_idx < n_out; out_idx++) {
            int c_start = out_idx * n_slots;
            int c_finis = c_start + n_slots; if (c_finis > n_cts) c_finis = n_cts;
            int n_range = c_finis - c_start;

            mpz_set_ui(out[out_idx], 1);
            for (int slot = n_range - 1; slot >= 0; slot--) {
                if (slot < n_range - 1)
                    mpz_powm(out[out_idx], out[out_idx], ctx->shift_pow, ctx->n_sq);
                mpz_mul(out[out_idx], out[out_idx], x[c_start + slot]);
                mpz_mod(out[out_idx], out[out_idx], ctx->n_sq);
            }
        }
        mpz_clear(bn);
    }

    for (int c = 0; c < n_cts; c++) mpz_clear(x[c]);
    free(x);
    free(a_prime);
    free(b_prime);
    return n_out;
}

void decompress_response(compress_ctx_t *ctx, mpz_t *answer, uint8_t *plain, int n_thrds)
{
    int n_slots = ctx->n_slots;
    int log_r   = ctx->log_r;
    int log_p   = ctx->log_p;
    int n_cts   = ctx->n_cts;
    uint32_t p  = 1u << log_p;

    mpz_t slot_mask, delta_r, half;
    mpz_inits(slot_mask, delta_r, half, NULL);
    mpz_set_ui(slot_mask, 1);
    mpz_mul_2exp(slot_mask, slot_mask, log_r);
    mpz_sub_ui(slot_mask, slot_mask, 1);
    mpz_set_ui(delta_r, 1);
    mpz_mul_2exp(delta_r, delta_r, log_r - log_p);
    mpz_fdiv_q_2exp(half, delta_r, 1);

    if (n_thrds < 1) n_thrds = 1;

    #pragma omp parallel num_threads(n_thrds)
    {
        mpz_t dec, slot_val;
        mpz_inits(dec, slot_val, NULL);

        #pragma omp for schedule(static)
        for (int out_idx = 0; out_idx < ctx->n_out; out_idx++) {
            paillier_decrypt(dec, ctx, answer[out_idx]);

            int c_start = out_idx * n_slots;
            int c_finis = c_start + n_slots; if (c_finis > n_cts) c_finis = n_cts;
            for (int slot = 0; slot < c_finis - c_start; slot++) {
                int c = c_start + slot;
                mpz_and(slot_val, dec, slot_mask);
                mpz_add(slot_val, slot_val, half);
                mpz_fdiv_q(slot_val, slot_val, delta_r);
                plain[c] = mpz_get_ui(slot_val) % p;
                mpz_fdiv_q_2exp(dec, dec, log_r);
            }
        }
        mpz_clears(dec, slot_val, NULL);
    }

    mpz_clears(slot_mask, delta_r, half, NULL);
}
