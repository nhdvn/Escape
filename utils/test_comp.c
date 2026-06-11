/* test_comp -- combined correctness + performance bench for the IFMA52 Mont
 * primitives (scalar + 16-way) AND the full Paillier-based compress
 * pipeline.  Run back-to-back:
 *
 *   [1] Mont modmul:
 *         (a) GMP  mpz_mul+mpz_mod                       -- baseline
 *         (b) mont_n80_modmul             (scalar CIOS)      -- precompute path
 *         (c) mont_n80_w16_modmul   (16 ct's per call) -- compress hot path
 *       Each runs a long chain c <- c*b[i] mod m.  The 16-way reports both
 *       per-call us and per-ct us (16 ct's processed per call).
 *
 *   [2] Compress pipeline:
 *         compress_response + decompress_response on synthetic LWE inputs.
 *       Verifies the plaintext round-trip across n_cts chunks.
 */
#include "compress.h"
#include "mont_n80.h"
#include "measure.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>  /* __m512i / _mm512_* used in Section [1] 16-way bench */

#ifdef __AVX512IFMA__
#define MONT_BENCH_OPS 200000
#define R_BITS         (MONT_N80 * MONT_LIMB_BITS)
#endif

/* ===================================================================
 * Section [1]: Mont modmul correctness + perf  (IFMA-only)
 * =================================================================== */
#ifdef __AVX512IFMA__
static int run_mont_section(void)
{
    printf("==== [1] Mont modmul (chain of %d) ====\n", MONT_BENCH_OPS);

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 0xc0ffeeUL);

    mpz_t p, q, N_pq, m, R, R_N, N_exp;
    mpz_init(p);
    mpz_init(q);
    mpz_init(N_pq);
    mpz_init(m);
    mpz_init_set_ui(R, 1);
    mpz_mul_2exp(R, R, R_BITS);
    mpz_init(R_N);
    mpz_init_set_ui(N_exp, MONT_BENCH_OPS);

    mpz_urandomb(p, rs, 1024); mpz_setbit(p, 1023); mpz_nextprime(p, p);
    mpz_urandomb(q, rs, 1024); mpz_setbit(q, 1023); mpz_nextprime(q, q);
    mpz_mul(N_pq, p, q);
    mpz_mul(m, N_pq, N_pq);
    mpz_clear(p); mpz_clear(q); mpz_clear(N_pq);
    mpz_powm(R_N, R, N_exp, m);
    mpz_clear(N_exp);

    printf("  modulus bits: %zu\n", mpz_sizeinbase(m, 2));

    mpz_t *bb = malloc(MONT_BENCH_OPS * sizeof(mpz_t));
    for (int i = 0; i < MONT_BENCH_OPS; i++) {
        mpz_init(bb[i]);
        mpz_urandomm(bb[i], rs, m);
    }
    mpz_t c_init_z;
    mpz_init(c_init_z);
    mpz_urandomm(c_init_z, rs, m);

    uint64_t m_l[MONT_N80];
    mont_n80_mpz_to_limbs52(m_l, m);
    uint64_t m_inv_neg = mont_compute_m_inv_neg(m_l);

    uint64_t (*b_l)[MONT_N80] = malloc(MONT_BENCH_OPS * sizeof *b_l);
    for (int i = 0; i < MONT_BENCH_OPS; i++) mont_n80_mpz_to_limbs52(b_l[i], bb[i]);

    uint64_t c_init_l[MONT_N80];
    mont_n80_mpz_to_limbs52(c_init_l, c_init_z);

    /* GMP chain. */
    mpz_t c_gmp, tmp;
    mpz_init_set(c_gmp, c_init_z);
    mpz_init(tmp);
    uint64_t t0 = now_ns();
    for (int i = 0; i < MONT_BENCH_OPS; i++) {
        mpz_mul(tmp, c_gmp, bb[i]);
        mpz_mod(c_gmp, tmp, m);
    }
    double gmp_time = (now_ns() - t0) / 1e9;
    mpz_clear(tmp);

    /* Mont AVX scalar chain. */
    uint64_t c_avx[MONT_N80];
    memcpy(c_avx, c_init_l, sizeof c_avx);
    t0 = now_ns();
    for (int i = 0; i < MONT_BENCH_OPS; i++) {
        mont_n80_modmul(c_avx, c_avx, b_l[i], m_l, m_inv_neg);
    }
    double avx_time = (now_ns() - t0) / 1e9;

    /* Correctness: c_avx * R^N mod m  ==  c_gmp. */
    mpz_t avx_check;
    mpz_init(avx_check);
    mont_n80_limbs52_to_mpz(avx_check, c_avx);
    mpz_mul(avx_check, avx_check, R_N);
    mpz_mod(avx_check, avx_check, m);
    int ok_scalar = (mpz_cmp(avx_check, c_gmp) == 0);

    /* 16-way mont_n80_w16_modmul chain: 16 cts in parallel via 2 instances.
     * Each ct has its own (a, b) pair.  Replicate scalar b across all 16
     * lanes so lane 0 reproduces the scalar chain (correctness target). */
    enum { N_VEC = MONT_N80 };
    __m512i c_v_a[N_VEC], c_v_b[N_VEC];
    __m512i b_v_a[N_VEC], b_v_b[N_VEC];
    for (int k = 0; k < N_VEC; k++) {
        c_v_a[k] = _mm512_set1_epi64((long long)c_init_l[k]);
        c_v_b[k] = _mm512_set1_epi64((long long)c_init_l[k]);
    }
    t0 = now_ns();
    for (int i = 0; i < MONT_BENCH_OPS; i++) {
        for (int k = 0; k < N_VEC; k++) {
            b_v_a[k] = _mm512_set1_epi64((long long)b_l[i][k]);
            b_v_b[k] = _mm512_set1_epi64((long long)b_l[i][k]);
        }
        mont_n80_w16_modmul(c_v_a, c_v_b, c_v_a, b_v_a, c_v_b, b_v_b,
                              m_l, m_inv_neg);
    }
    double v16_time = (now_ns() - t0) / 1e9;

    /* Correctness: lane 0 of c_v_a == c_avx (same scalar chain). */
    uint64_t c_v_lane0[MONT_N80];
    for (int k = 0; k < MONT_N80; k++) {
        uint64_t lanes[8] __attribute__((aligned(64)));
        _mm512_store_si512((__m512i *)lanes, c_v_a[k]);
        c_v_lane0[k] = lanes[0];
    }
    int ok_v16 = (memcmp(c_v_lane0, c_avx, sizeof c_avx) == 0);

    int ok = ok_scalar && ok_v16;
    printf("  correctness: %s  (scalar=%s, 16-way=%s)\n",
           ok ? "PASS" : "FAIL",
           ok_scalar ? "PASS" : "FAIL", ok_v16 ? "PASS" : "FAIL");
    printf("  GMP   mpz_mul+mpz_mod  : %6.3f us/op  (%7.0f ops/s)  vs GMP %.2fx\n",
           gmp_time / MONT_BENCH_OPS * 1e6, MONT_BENCH_OPS / gmp_time, 1.0);
    printf("  Mont  scalar (CIOS)    : %6.3f us/op  (%7.0f ops/s)  vs GMP %.2fx\n",
           avx_time / MONT_BENCH_OPS * 1e6, MONT_BENCH_OPS / avx_time,
           gmp_time / avx_time);
    printf("  Mont  16way (3kara x2) : %6.3f us/call (%6.3f us/ct)  vs GMP %.2fx per ct\n",
           v16_time / MONT_BENCH_OPS * 1e6,
           v16_time / MONT_BENCH_OPS * 1e6 / 16.0,
           gmp_time / (v16_time / 16.0));

    mpz_clear(avx_check);
    mpz_clear(c_init_z);
    mpz_clear(c_gmp);
    mpz_clear(R);
    mpz_clear(R_N);
    mpz_clear(m);
    for (int i = 0; i < MONT_BENCH_OPS; i++) mpz_clear(bb[i]);
    free(bb);
    free(b_l);
    gmp_randclear(rs);
    return ok ? 0 : 1;
}
#endif /* __AVX512IFMA__ */

/* ===================================================================
 * Section [2]: Compress pipeline (synthetic LWE -> compress -> decompress)
 * =================================================================== */
static uint32_t xrand32(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*state >> 32);
}

static void make_lwe_cts(const compress_ctx_t *ctx,
                         uint64_t *rng,
                         uint32_t *lwe_a,
                         uint32_t *lwe_b,
                         uint8_t  *plain_orig)
{
    int      lmbda  = ctx->lmbda;
    int      n_cts  = ctx->n_cts;
    uint32_t Delta  = (ctx->log_q - ctx->log_p >= 32) ? 0u
                                                      : (1u << (ctx->log_q - ctx->log_p));
    uint32_t p_mask = (1u << ctx->log_p) - 1u;

    for (int c = 0; c < n_cts; c++) {
        plain_orig[c] = (uint8_t)(xrand32(rng) & p_mask);
        uint32_t inner = 0;
        for (int i = 0; i < lmbda; i++) {
            uint32_t a_ci = xrand32(rng);
            lwe_a[(size_t)c * lmbda + i] = a_ci;
            inner += a_ci * (uint32_t)ctx->sk[i];
        }
        int32_t e_c = (int32_t)(xrand32(rng) % 11) - 5;
        lwe_b[c] = inner + (uint32_t)e_c + Delta * (uint32_t)plain_orig[c];
    }
}

static int run_compress_section(int lmbda, int n_cts, int n_thrds, int c_thrds)
{
    int log_q   = 32;
    int log_p   = 8;
    int pl_bits = 2048;

    omp_set_num_threads(n_thrds);

    printf("\n==== [2] Compress pipeline ====\n");
    printf("  lmbda=%d  n_cts=%d  log_q=%d  log_p=%d  pl_bits=%d  threads=%d\n",
           lmbda, n_cts, log_q, log_p, pl_bits, n_thrds);

    uint64_t t0 = now_ns();
    compress_ctx_t *ctx = compress_ctx_setup(lmbda, n_cts, log_q, log_p, pl_bits);
    double t_keygen = (now_ns() - t0) / 1e9;

    t0 = now_ns();
    precompute_random(ctx, c_thrds);
    double t_pool = (now_ns() - t0) / 1e9;

    t0 = now_ns();
    compress_key(ctx, c_thrds);
    double t_zkey = (now_ns() - t0) / 1e9;

    t0 = now_ns();
    precompute_window(ctx);
    double t_zkw = (now_ns() - t0) / 1e9;

    size_t out_bytes = (size_t)ctx->n_out * (pl_bits * 2 / 8);
    printf("  log_r=%d  slots/ct=%d  output_cts=%d (%.1f KiB)\n",
           ctx->log_r, ctx->n_slots, ctx->n_out, out_bytes / 1024.0);
    printf("  keygen : %.3f s  [one-time per context]\n", t_keygen);
    printf("  r_pool : %.3f s  [offline, between queries]\n", t_pool);
    printf("  zkey   : %.3f s  [per query, client]\n", t_zkey);
    printf("  zkw    : %.3f s  [per query, server]\n", t_zkw);

    uint64_t  rng = 0xdeadbeefcafebabeULL;
    uint32_t *lwe_a      = malloc((size_t)n_cts * lmbda * sizeof(uint32_t));
    uint32_t *lwe_b      = malloc((size_t)n_cts * sizeof(uint32_t));
    uint8_t  *plain_orig = malloc((size_t)n_cts);
    uint8_t  *plain_recv = malloc((size_t)n_cts);
    make_lwe_cts(ctx, &rng, lwe_a, lwe_b, plain_orig);

    mpz_t *out = malloc((size_t)ctx->n_out * sizeof(mpz_t));
    for (int i = 0; i < ctx->n_out; i++) mpz_init(out[i]);

    t0 = now_ns();
    compress_response(ctx, lwe_a, lwe_b, out);
    double t_comp = (now_ns() - t0) / 1e9;
    printf("  compress  : %.2f s  (%.2f ms / chunk  |  %.2f s / output-ct)\n",
           t_comp, t_comp * 1e3 / n_cts, t_comp / ctx->n_out);

    t0 = now_ns();
    decompress_response(ctx, out, plain_recv, c_thrds);
    double t_dec = (now_ns() - t0) / 1e9;
    printf("  decompress: %.3f s\n", t_dec);

    int errors = 0, first_err = -1;
    for (int c = 0; c < n_cts; c++) {
        if (plain_recv[c] != plain_orig[c]) {
            errors++;
            if (first_err < 0) first_err = c;
        }
    }
    if (errors == 0) {
        printf("  correctness: PASS  (%d / %d chunks match)\n", n_cts, n_cts);
    } else {
        printf("  correctness: FAIL  (%d / %d errors; first at chunk %d: orig=%u recv=%u)\n",
               errors, n_cts, first_err,
               (unsigned)plain_orig[first_err], (unsigned)plain_recv[first_err]);
    }

    for (int i = 0; i < ctx->n_out; i++) mpz_clear(out[i]);
    free(out);
    free(plain_recv);
    free(plain_orig);
    free(lwe_a);
    free(lwe_b);
    compress_ctx_free(ctx);
    return errors ? 1 : 0;
}

int main(int argc, char **argv)
{
    int lmbda   = (argc > 1) ? atoi(argv[1]) : 1024;
    int n_cts   = (argc > 2) ? atoi(argv[2]) : 4096;
    int n_thrds = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    int c_thrds = (argc > 4) ? atoi(argv[4]) : 3;

#ifdef __AVX512IFMA__
    int rc1 = run_mont_section();
#else
    printf("==== [1] Mont modmul: SKIPPED (built without __AVX512IFMA__) ====\n");
    int rc1 = 0;
#endif
    int rc2 = run_compress_section(lmbda, n_cts, n_thrds, c_thrds);
    return (rc1 || rc2) ? 1 : 0;
}
