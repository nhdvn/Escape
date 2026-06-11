/* _m512.c (AVX-512-IFMA52) override of _gmpz.c. Stays in Mont form across
 * the per-c hot loop and the slot-pack stage. Converts to mpz_t once per
 * output ct. Shared helpers (mod_switch, paillier_keygen) in compress.h. */

#include "compress.h"
#include "mont_n80.h"
#include "mont_n40.h"
#include "../utils/measure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Pack 8 lane-pointers' k-th limbs into out_v. */
#define LANE_PACK_8(out_v, k_, p0, p1, p2, p3, p4, p5, p6, p7)                \
    ({ (out_v) = _mm512_set_epi64(                                            \
            (long long)(p7)[k_], (long long)(p6)[k_],                         \
            (long long)(p5)[k_], (long long)(p4)[k_],                         \
            (long long)(p3)[k_], (long long)(p2)[k_],                         \
            (long long)(p1)[k_], (long long)(p0)[k_]); })

/* Pack 8 consecutive rows of a 2D buf (buf[off..off+7][k]) into out_v. */
#define LANE_PACK_8_2D(out_v, k_, buf_, off_)                                   \
    LANE_PACK_8((out_v), (k_),                                                  \
        (buf_)[(off_)+0], (buf_)[(off_)+1], (buf_)[(off_)+2], (buf_)[(off_)+3], \
        (buf_)[(off_)+4], (buf_)[(off_)+5], (buf_)[(off_)+6], (buf_)[(off_)+7])

static void paillier_decrypt(mpz_t out, const compress_ctx_t *ctx, const mpz_t c)
{
    mpz_t t; mpz_init(t);
    uint64_t c_l[MONT_N80], t_l[MONT_N80];
    mont_n80_mpz_to_limbs52(c_l, c);
    mont_n80_modexp(t_l, c_l, ctx->d_xp, ctx->n_sq_l, ctx->R2_l, ctx->m_inv_neg);
    mont_n80_limbs52_to_mpz(t, t_l);
    mpz_sub_ui(t, t, 1);
    mpz_divexact(t, t, ctx->n);
    mpz_mul(out, t, ctx->mu);
    mpz_mod(out, out, ctx->n);
    mpz_clear(t);
}

/* ---- Context lifecycle ------------------------------------------------- */

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

    mpz_inits(ctx->n, ctx->n_sq, ctx->d_xp, ctx->mu, ctx->shift_pow, ctx->p, ctx->q, 
        ctx->p_sq, ctx->q_sq, ctx->exp_p, ctx->exp_q, ctx->h_p, ctx->h_q, ctx->p_inv_q, NULL);

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 0xdeadcafeUL);
    paillier_keygen(ctx, rs);
    gmp_randclear(rs);

    mpz_set_ui(ctx->shift_pow, 1);
    mpz_mul_2exp(ctx->shift_pow, ctx->shift_pow, (mp_bitcnt_t)ctx->log_r);

    /* Full-size Mont state: n_sq, m_inv_neg, R^2 mod n_sq. */
    mont_n80_mpz_to_limbs52(ctx->n_sq_l, ctx->n_sq);
    ctx->m_inv_neg = mont_compute_m_inv_neg(ctx->n_sq_l);
    mpz_t R2;
    mpz_init_set_ui(R2, 1);
    mpz_mul_2exp(R2, R2, (mp_bitcnt_t)(2 * MONT_N80 * MONT_LIMB_BITS));
    mpz_mod(R2, R2, ctx->n_sq);
    mont_n80_mpz_to_limbs52(ctx->R2_l, R2);

    /* Half-size Mont state for CRT (mod p^2 and mod q^2).  R_half = 2^(40*52). */
    mont_n40_mpz_to_limbs52(ctx->p_sq_l, ctx->p_sq);
    mont_n40_mpz_to_limbs52(ctx->q_sq_l, ctx->q_sq);
    ctx->m_inv_neg_p = mont_compute_m_inv_neg(ctx->p_sq_l);
    ctx->m_inv_neg_q = mont_compute_m_inv_neg(ctx->q_sq_l);
    
    mpz_set_ui(R2, 1);
    mpz_mul_2exp(R2, R2, (mp_bitcnt_t)(2 * MONT_N40 * MONT_LIMB_BITS));
    mpz_t Rh; mpz_init(Rh);

    mpz_mod(Rh, R2, ctx->p_sq); mont_n40_mpz_to_limbs52(ctx->R2_p_l, Rh);
    mpz_mod(Rh, R2, ctx->q_sq); mont_n40_mpz_to_limbs52(ctx->R2_q_l, Rh);

    mpz_clear(Rh);
    mpz_clear(R2);
    return ctx;
}

void compress_ctx_free(compress_ctx_t *ctx)
{
    if (!ctx) return;
    mpz_clears(ctx->n, ctx->n_sq, ctx->d_xp, ctx->mu, ctx->shift_pow, ctx->p, ctx->q, 
        ctx->p_sq, ctx->q_sq, ctx->exp_p, ctx->exp_q, ctx->h_p, ctx->h_q, ctx->p_inv_q, NULL);
    
    if (ctx->zkey) {
        for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->zkey[i]);
        free(ctx->zkey);
    }
    if (ctx->zkw_mont) free(ctx->zkw_mont);
    if (ctx->r_pow_n) {
        for (int i = 0; i < ctx->lmbda; i++) mpz_clear(ctx->r_pow_n[i]);
        free(ctx->r_pow_n);
    }
    free(ctx->sk);
    free(ctx);
}

/* ---- Pre-computation (per-query) -------------------------------------- */

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
            uint64_t r_l[MONT_N80], rn_l[MONT_N80];
            mont_n80_mpz_to_limbs52(r_l, r);
            mont_n80_modexp(rn_l, r_l, ctx->n, ctx->n_sq_l, ctx->R2_l, ctx->m_inv_neg);
            mont_n80_limbs52_to_mpz(ctx->r_pow_n[i], rn_l);
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
    gmp_randseed_ui(rs, 0xfeed);
    for (int i = 0; i < lmbda; i++) {
        ctx->sk[i] = (uint8_t)(gmp_urandomb_ui(rs, 1));
        mpz_init(ctx->zkey[i]);
    }
    gmp_randclear(rs);

    #pragma omp parallel num_threads(n_thrds)
    {
        mpz_t bn; mpz_init(bn);
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
    if (ctx->zkw_mont) { free(ctx->zkw_mont); ctx->zkw_mont = NULL; }

    ctx->zkw_bits   = ZKW;
    ctx->zkn_digits = (ctx->log_r + ZKW - 1) / ZKW;
    int n_digits  = ctx->zkn_digits;
    int n_entries = (1 << ZKW) - 1;

    /* Table built directly in Mont form (no mpz_t intermediate). Single
     * fork-join over i: each thread runs its slice's seed + squaring chain
     * + multiply chain serially -- avoids per-(d,k) fork-join overhead. */
    size_t mont_total = (size_t)n_digits * n_entries * lmbda;
    ctx->zkw_mont = malloc(mont_total * MONT_N80 * sizeof(uint64_t));

    #define ZKW_MONT(d, k, i) \
        (ctx->zkw_mont + (((size_t)(d) * lmbda + (i)) * n_entries + (k)) * MONT_N80)

    int n_batch = lmbda & ~15;                      /* main loop: 16 i's at a time */
    size_t i_stride = (size_t)n_entries * MONT_N80;   /* ZKW_MONT i-stride in u64s */

    #pragma omp parallel for schedule(static)
    for (int i_batch = 0; i_batch < n_batch; i_batch += 16) {
        /* Pack 16 zkeys to (zkey_a, zkey_b) lane vectors. */
        uint64_t zkey_l[16][MONT_N80];
        for (int lane = 0; lane < 16; lane++)
            mont_n80_mpz_to_limbs52(zkey_l[lane], ctx->zkey[i_batch + lane]);

        __m512i zkey_a[MONT_N80], zkey_b[MONT_N80];
        for (int k = 0; k < MONT_N80; k++) {
            LANE_PACK_8_2D(zkey_a[k], k, zkey_l, 0);
            LANE_PACK_8_2D(zkey_b[k], k, zkey_l, 8);
        }

        /* R2 broadcast into vector form. */
        __m512i R2_v[MONT_N80];
        for (int k = 0; k < MONT_N80; k++)
            R2_v[k] = _mm512_set1_epi64((long long)ctx->R2_l[k]);

        /* base[d] = Mont(zkey^(2^(d*ZKW))) -- per-lane independent chain. */
        __m512i inp_a[n_digits][MONT_N80], inp_b[n_digits][MONT_N80];

        /* base[0] = Mont(zkey) = zkey * R2 / R. */
        mont_n80_w16_modmul(inp_a[0], inp_b[0], zkey_a, R2_v, zkey_b, 
                            R2_v, ctx->n_sq_l, ctx->m_inv_neg);

        /* base[d] = base[d-1]^(2^ZKW). */
        for (int d = 1; d < n_digits; d++) {
            for (int k = 0; k < MONT_N80; k++) {
                inp_a[d][k] = inp_a[d-1][k];
                inp_b[d][k] = inp_b[d-1][k];
            }
            for (int s = 0; s < ZKW; s++)
                mont_n80_w16_modmul(inp_a[d], inp_b[d], inp_a[d], inp_a[d],
                            inp_b[d], inp_b[d], ctx->n_sq_l, ctx->m_inv_neg);
        }

        /* Helper: scatter 16-lane (a,b) result to ZKW_MONT(d, k, i_batch + lane). */
        #define ZKW16_STORE(D, K, A, B) ({                                      \
            uint64_t _buf[MONT_N80][16] __attribute__((aligned(64)));           \
            for (int _kk = 0; _kk < MONT_N80; _kk++) {                          \
                _mm512_store_si512((__m512i *)&_buf[_kk][0], (A)[_kk]);         \
                _mm512_store_si512((__m512i *)&_buf[_kk][8], (B)[_kk]);         \
            }                                                                   \
            for (int _l = 0; _l < 16; _l++) {                                   \
                uint64_t *_dst = ZKW_MONT((D), (K), 0) + (size_t)(i_batch + _l) * i_stride; \
                for (int _kk = 0; _kk < MONT_N80; _kk++)                        \
                    _dst[_kk] = _buf[_kk][_l];                                  \
            }                                                                   \
        })

        /* Write base[d] into zkw[d][0][i_batch + lane] for each d, lane. */
        for (int d = 0; d < n_digits; d++)
            ZKW16_STORE(d, 0, inp_a[d], inp_b[d]);

        /* zkw[d][k] = zkw[d][k-1] * base[d] for k = 1..n_entries-1. */
        for (int d = 0; d < n_digits; d++) {
            __m512i cur_a[MONT_N80], cur_b[MONT_N80];
            for (int k = 0; k < MONT_N80; k++) {
                cur_a[k] = inp_a[d][k];
                cur_b[k] = inp_b[d][k];
            }
            for (int kk = 1; kk < n_entries; kk++) {
                mont_n80_w16_modmul(cur_a, cur_b, cur_a, inp_a[d],
                            cur_b, inp_b[d], ctx->n_sq_l, ctx->m_inv_neg);
                ZKW16_STORE(d, kk, cur_a, cur_b);
            }
        }

        #undef ZKW16_STORE
    }

    /* Tail: scalar path for any leftover i's. */
    #pragma omp parallel for schedule(static)
    for (int i = n_batch; i < lmbda; i++) {
        uint64_t zkey_l[MONT_N80];
        mont_n80_mpz_to_limbs52(zkey_l, ctx->zkey[i]);
        mont_n80_modmul(ZKW_MONT(0, 0, i), zkey_l, ctx->R2_l,
                    ctx->n_sq_l, ctx->m_inv_neg);

        for (int d = 1; d < n_digits; d++) {
            uint64_t tmp[MONT_N80];
            memcpy(tmp, ZKW_MONT(d - 1, 0, i), MONT_N80 * sizeof(uint64_t));
            for (int s = 0; s < ZKW; s++)
                mont_n80_modmul(tmp, tmp, tmp, ctx->n_sq_l, ctx->m_inv_neg);
            memcpy(ZKW_MONT(d, 0, i), tmp, MONT_N80 * sizeof(uint64_t));
        }

        for (int d = 0; d < n_digits; d++)
            for (int k = 1; k < n_entries; k++)
                mont_n80_modmul(ZKW_MONT(d, k, i), ZKW_MONT(d, k - 1, i),
                            ZKW_MONT(d, 0, i), ctx->n_sq_l, ctx->m_inv_neg);
    }

    #undef ZKW_MONT
}

/* ---- Online compress / decompress -------------------------------------
 *  Single fused parallel region (init / hot loop / tail / slot-pack).
 *  Hot loop batches 16 cts per mont_n80_w16_modmul call.
 *  LANE_PACK_8 builds __m512i lanes directly from per-ct pointers
 *  (no scratch buffer); ZKW_PTR fuses the digit==0 -> Mont(1) shortcut
 *  with the table lookup so mont_one stays register-resident. */

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
    const int n_ent   = (1 << ZKW) - 1;

    uint32_t *a_prime = malloc((size_t)n_cts * lmbda * sizeof(uint32_t));
    uint32_t *b_prime = malloc((size_t)n_cts * sizeof(uint32_t));
    mod_switch_u32(a_prime, vA, (size_t)n_cts * lmbda, ctx->log_q, log_r);
    mod_switch_u32(b_prime, vb, (size_t)n_cts,         ctx->log_q, log_r);

    uint64_t (*x_mont)[MONT_N80] = malloc((size_t)n_cts * sizeof(uint64_t[MONT_N80]));

    /* Mont(1) = R mod n_sq -- no-op multiplier when a digit is 0. */
    uint64_t one_l[MONT_N80] = { 1 };
    uint64_t mont_one[MONT_N80];
    mont_n80_modmul(mont_one, one_l, ctx->R2_l, ctx->n_sq_l, ctx->m_inv_neg);

    int n_batch = n_cts & ~15;

    /* dig = 0 is the implicit "*Mont(1)" shortcut; nonzero digs index the
     * pre-built window table at row (d, i), entry (dig - 1). */
    #define ZKW_OFF(d_, dig_, i_) \
        ((((size_t)(d_) * lmbda + (i_)) * n_ent + (dig_) - 1) * MONT_N80)
    #define ZKW_PTR(d_, dig_, i_) \
        ((dig_) == 0 ? mont_one : ctx->zkw_mont + ZKW_OFF(d_, dig_, i_))

    int max_t = omp_get_max_threads();
    long     *per_total  = calloc(max_t, sizeof(long));
    uint64_t *per_cycle  = calloc(max_t, sizeof(uint64_t));
    uint64_t *per_stall  = calloc(max_t, sizeof(uint64_t));
    uint64_t *per_memory = calloc(max_t, sizeof(uint64_t));
    uint64_t t_comp0  = now_ns();

    #pragma omp parallel
    {
        long delay_total = 0;
        uint64_t delay_cycle = 0, delay_stall = 0, delay_memory = 0;
        measure_t pc = MEASURE_INIT;
        int pc_ok = (measure_init(&pc) == 0);

        MEASURE_START();

        /* Step 1: per-c init (to-Mont of (1 + b'[c]*N) mod n_sq). */
        mpz_t init;
        mpz_init(init);
        #pragma omp for schedule(static)
        for (int c = 0; c < n_cts; c++) {
            mpz_set_ui(init, b_prime[c]);
            mpz_mul(init, init, ctx->n);
            mpz_add_ui(init, init, 1);
            mpz_mod(init, init, ctx->n_sq);
            uint64_t x_l[MONT_N80];
            mont_n80_mpz_to_limbs52(x_l, init);
            mont_n80_modmul(x_mont[c], x_l, ctx->R2_l, ctx->n_sq_l, ctx->m_inv_neg);
        }
        mpz_clear(init);

        /* Step 2: hot loop -- 16 cts per call to mont_n80_w16_modmul. */
        #pragma omp for schedule(static)
        for (int c_base = 0; c_base < n_batch; c_base += 16) {
            __m512i x_a[MONT_N80], x_b[MONT_N80];
            for (int k = 0; k < MONT_N80; k++) {
                LANE_PACK_8(x_a[k], k,
                    x_mont[c_base+0], x_mont[c_base+1], x_mont[c_base+2], x_mont[c_base+3],
                    x_mont[c_base+4], x_mont[c_base+5], x_mont[c_base+6], x_mont[c_base+7]);
                LANE_PACK_8(x_b[k], k,
                    x_mont[c_base+8],  x_mont[c_base+9],  x_mont[c_base+10], x_mont[c_base+11],
                    x_mont[c_base+12], x_mont[c_base+13], x_mont[c_base+14], x_mont[c_base+15]);
            }

            for (int i = 0; i < lmbda; i++) {
                uint32_t scal[16];
                for (int lane = 0; lane < 16; lane++)
                    scal[lane] = ((uint32_t)0 - a_prime[(size_t)(c_base+lane) * lmbda + i]) & r_mask;

                for (int d = 0; d < n_digits; d++) {
                    const uint64_t *p[16];
                    for (int lane = 0; lane < 16; lane++)
                        p[lane] = ZKW_PTR(d, ((scal[lane] >> (d * ZKW)) & w_mask), i);

                    __m512i zkw_a[MONT_N80], zkw_b[MONT_N80];
                    for (int k = 0; k < MONT_N80; k++) {
                        LANE_PACK_8(zkw_a[k], k, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                        LANE_PACK_8(zkw_b[k], k, p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
                    }
                    mont_n80_w16_modmul(x_a, x_b, x_a, zkw_a, x_b, zkw_b, ctx->n_sq_l, ctx->m_inv_neg);
                }
            }

            for (int k = 0; k < MONT_N80; k++) {
                uint64_t buf_a[8] __attribute__((aligned(64)));
                uint64_t buf_b[8] __attribute__((aligned(64)));
                _mm512_store_si512((__m512i *)buf_a, x_a[k]);
                _mm512_store_si512((__m512i *)buf_b, x_b[k]);
                for (int lane = 0; lane < 8; lane++) {
                    x_mont[c_base + lane    ][k] = buf_a[lane];
                    x_mont[c_base + 8 + lane][k] = buf_b[lane];
                }
            }
        }

        /* Step 3: tail (n_cts % 16) -- scalar fallback. */
        #pragma omp for schedule(static)
        for (int c = n_batch; c < n_cts; c++) {
            const uint32_t *a_row = a_prime + (size_t)c * lmbda;
            for (int i = 0; i < lmbda; i++) {
                uint32_t scal = ((uint32_t)0 - a_row[i]) & r_mask;
                for (int d = 0; d < n_digits && scal; d++, scal >>= ZKW) {
                    uint32_t digit = scal & w_mask;
                    if (digit == 0) continue;
                    const uint64_t *zkw = ZKW_PTR(d, digit, i);
                    mont_n80_modmul(x_mont[c], x_mont[c], zkw, ctx->n_sq_l, ctx->m_inv_neg);
                }
            }
        }

        /* Slot-pack: per out_idx, res = sum x[c] * (2^log_r)^(n_range-1-slot).
         * High-to-low: log_r explicit Mont squarings replace one mpz_powm. */
        #pragma omp for schedule(static)
        for (int out_idx = 0; out_idx < n_out; out_idx++) {
            int c_start = out_idx * n_slots;
            int c_finis = c_start + n_slots; if (c_finis > n_cts) c_finis = n_cts;
            int n_range = c_finis - c_start;

            uint64_t res[MONT_N80];
            memcpy(res, mont_one, sizeof(res));
            for (int slot = n_range - 1; slot >= 0; slot--) {
                if (slot < n_range - 1)
                    for (int s = 0; s < log_r; s++)
                        mont_n80_modmul(res, res, res, ctx->n_sq_l, ctx->m_inv_neg);
                mont_n80_modmul(res, res, x_mont[c_start + slot],
                            ctx->n_sq_l, ctx->m_inv_neg);
            }
            uint64_t res_natural[MONT_N80];
            mont_n80_modmul(res_natural, res, one_l, ctx->n_sq_l, ctx->m_inv_neg);
            mont_n80_limbs52_to_mpz(out[out_idx], res_natural);
        }

        MEASURE_FINIS();

        int tid = omp_get_thread_num();
        per_total[tid]  = delay_total;
        per_cycle[tid]  = delay_cycle;
        per_stall[tid]  = delay_stall;
        per_memory[tid] = delay_memory;

        if (pc_ok) measure_close(&pc);
    }

    double   e2e_total = (now_ns() - t_comp0) / 1e6;
    long     max_total = 0;
    uint64_t max_cycle = 0, max_stall = 0, max_memory = 0;
    for (int t = 0; t < max_t; t++) {
        if (per_total[t] > max_total) {
            max_total  = per_total[t];
            max_cycle = per_cycle[t];
            max_stall = per_stall[t];
            max_memory = per_memory[t];
        }
    }
    if (max_cycle > 0) {
        double freq = (double)max_cycle / ((double)max_total / 1e9);
        double cpu_total = max_total / 1e6;
        double cmp_cycle = (double)(max_cycle - max_stall) / freq * 1e3;
        double cpu_stall = (double)(max_stall - max_memory) / freq * 1e3;
        double mem_stall = (double)(max_memory)            / freq * 1e3;
        printf("  [server cmp]   %.1f ms\n", e2e_total);
        printf("    cmp_cycle    %.1f ms\n", cmp_cycle);
        printf("    cpu_stall    %.1f ms\n", cpu_stall);
        printf("    mem_stall    %.1f ms  (%.0f%% of compress)\n",
               mem_stall, 100.0 * mem_stall / cpu_total);
    } else {
        printf("  [compress]     %.1f ms\n", e2e_total);
    }
    free(per_total); free(per_cycle); free(per_stall); free(per_memory);

    #undef ZKW_PTR
    #undef ZKW_OFF

    free(x_mont);
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
    int n_batch = ctx->n_out & ~15;     /* multiple of 16 */

    /* Paillier-CRT: replace c^lambda mod N^2 with c^(p-1) mod p^2 and
     * c^(q-1) mod q^2 (both modulus and Carmichael exponent half-size,
     * 4x per modexp and 2x total). CRT-recombine per lane to m mod N. */
    #pragma omp parallel num_threads(n_thrds)
    {
        mpz_t dec, slot_val, tmp_p, tmp_q, mp_v, mq_v, recomb;
        mpz_inits(dec, slot_val, tmp_p, tmp_q, mp_v, mq_v, recomb, NULL);

        #pragma omp for schedule(static)
        for (int batch = 0; batch < n_batch; batch += 16) {
            /* Reduce each ciphertext mod p^2 / q^2 and pack 16 lanes. */
            uint64_t cp_l[16][MONT_N40], cq_l[16][MONT_N40];
            mpz_t cp, cq;
            mpz_inits(cp, cq, NULL);
            for (int lane = 0; lane < 16; lane++) {
                mpz_mod(cp, answer[batch + lane], ctx->p_sq);
                mpz_mod(cq, answer[batch + lane], ctx->q_sq);
                mont_n40_mpz_to_limbs52(cp_l[lane], cp);
                mont_n40_mpz_to_limbs52(cq_l[lane], cq);
            }
            mpz_clears(cp, cq, NULL);

            __m512i cp_a[MONT_N40], cp_b[MONT_N40];
            __m512i cq_a[MONT_N40], cq_b[MONT_N40];
            for (int k = 0; k < MONT_N40; k++) {
                LANE_PACK_8_2D(cp_a[k], k, cp_l, 0);
                LANE_PACK_8_2D(cp_b[k], k, cp_l, 8);
                LANE_PACK_8_2D(cq_a[k], k, cq_l, 0);
                LANE_PACK_8_2D(cq_b[k], k, cq_l, 8);
            }

            /* Two half-size 16-way modexps with Carmichael-reduced exponents. */
            __m512i tp_a[MONT_N40], tp_b[MONT_N40];
            __m512i tq_a[MONT_N40], tq_b[MONT_N40];
            mont_n40_w16_modexp(tp_a, tp_b, cp_a, cp_b, ctx->exp_p,
                             ctx->p_sq_l, ctx->R2_p_l, ctx->m_inv_neg_p);
            mont_n40_w16_modexp(tq_a, tq_b, cq_a, cq_b, ctx->exp_q,
                             ctx->q_sq_l, ctx->R2_q_l, ctx->m_inv_neg_q);

            /* Unpack and finish CRT per lane. */
            uint64_t tp_lane[16][MONT_N40], tq_lane[16][MONT_N40];
            for (int k = 0; k < MONT_N40; k++) {
                uint64_t a_buf[8], b_buf[8];
                _mm512_storeu_si512((__m512i *)a_buf, tp_a[k]);
                _mm512_storeu_si512((__m512i *)b_buf, tp_b[k]);
                for (int l = 0; l < 8; l++) {
                    tp_lane[l][k]     = a_buf[l];
                    tp_lane[l + 8][k] = b_buf[l];
                }
                _mm512_storeu_si512((__m512i *)a_buf, tq_a[k]);
                _mm512_storeu_si512((__m512i *)b_buf, tq_b[k]);
                for (int l = 0; l < 8; l++) {
                    tq_lane[l][k]     = a_buf[l];
                    tq_lane[l + 8][k] = b_buf[l];
                }
            }

            for (int lane = 0; lane < 16; lane++) {
                /* m_p = L_p(c^(p-1) mod p^2) * h_p mod p */
                mont_n40_limbs52_to_mpz(tmp_p, tp_lane[lane]);
                mpz_sub_ui(tmp_p, tmp_p, 1);
                mpz_divexact(tmp_p, tmp_p, ctx->p);
                mpz_mul(mp_v, tmp_p, ctx->h_p);
                mpz_mod(mp_v, mp_v, ctx->p);

                /* m_q = L_q(c^(q-1) mod q^2) * h_q mod q */
                mont_n40_limbs52_to_mpz(tmp_q, tq_lane[lane]);
                mpz_sub_ui(tmp_q, tmp_q, 1);
                mpz_divexact(tmp_q, tmp_q, ctx->q);
                mpz_mul(mq_v, tmp_q, ctx->h_q);
                mpz_mod(mq_v, mq_v, ctx->q);

                /* CRT recombine: m = m_p + p * ((m_q - m_p) * p^(-1) mod q). */
                mpz_sub(recomb, mq_v, mp_v);
                mpz_mul(recomb, recomb, ctx->p_inv_q);
                mpz_mod(recomb, recomb, ctx->q);
                mpz_mul(recomb, recomb, ctx->p);
                mpz_add(dec, mp_v, recomb);

                int out_idx = batch + lane;
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
        }

        /* Tail: < 16 outputs left, scalar paillier_decrypt. */
        #pragma omp for schedule(static)
        for (int out_idx = n_batch; out_idx < ctx->n_out; out_idx++) {
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

        mpz_clears(dec, slot_val, tmp_p, tmp_q, mp_v, mq_v, recomb, NULL);
    }

    mpz_clears(slot_mask, delta_r, half, NULL);
}
