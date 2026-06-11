#ifndef COMPRESS_H
#define COMPRESS_H

/*
 * Paillier compression of LWE ciphertexts under a shared binary secret.
 *   INPUTS:  n_cts independent LWE ciphertexts under sk in {0,1}^lmbda:
 *              ct_c = (a_c in Z_q^lmbda,  b_c in Z_q),  c in [0, n_cts).
 *   OUTPUT:  n_out packed Paillier ciphertexts (mpz_t mod N^2),
 *              n_out = ceil(n_cts / n_slots).
 * Two interchangeable backend implementations (Makefile picks one):
 *   _gmpz.c  : portable GMP fallback (mpz_t arithmetic).
 *   _m512.c  : AVX-512 IFMA52. Stay as Mont form across hot loops.
 * Both expose the same set of public symbols below, so the rest of the
 * codebase is oblivious to which path was linked.
 */

#include <stdint.h>
#include <stddef.h>
#include <gmp.h>

#ifndef ZKW
#define ZKW 11
#endif

#ifndef PL_BITS
#define PL_BITS 2048
#endif

/* Mirrors mont.h's MONT_N80 (80 limbs of 52 bits = 4160 bits, holds N^2
 * for pl_bits=2048).  Defined here so the struct below has a complete layout
 * without needing to drag mont.h (and AVX-512 intrinsics) into TUs that take
 * the GMP fallback path. */
#define MONT_N80 80
#define MONT_N40 40

typedef struct compress_ctx_t {
    int    lmbda;           /* LWE secret dim                                 */
    int    n_cts;           /* number of LWE cts to compress per call         */
    int    log_q;           /* LWE ciphertext modulus bits                    */
    int    log_p;           /* LWE plaintext modulus bits                     */
    int    log_r;           /* mod-switch target                              */
    int    n_slots;         /* phases packed per Paillier ciphertext          */
    int    pl_bits;         /* Paillier modulus bits                          */
    int    n_out;           /* ceil(n_cts / n_slots)                          */

    mpz_t  n;               /* Paillier public modulus                        */
    mpz_t  n_sq;            /* N^2                                            */
    mpz_t  d_xp;            /* private decryption exponent                    */
    mpz_t  mu;              /* private decryption multiplier                  */
    mpz_t  shift_pow;       /* 2^log_r                                        */

    /* CRT (Paillier-CRT) decryption helpers.  All half-size mpz state. */
    mpz_t  p, q;            /* prime factors of N                             */
    mpz_t  p_sq, q_sq;      /* p^2, q^2                                       */
    mpz_t  exp_p, exp_q;    /* p-1, q-1 (Carmichael-reduced exponents)        */
    mpz_t  h_p, h_q;        /* L_x(g^(x-1) mod x^2)^(-1) mod x, x in {p,q}    */
    mpz_t  p_inv_q;         /* p^(-1) mod q (for CRT recombine)               */

    uint8_t *sk;            /* [LWE lmbda] in {0,1}                           */
    mpz_t   *zkey;          /* [LWE lmbda] Paillier_Enc(sk[i])                */
    mpz_t   *r_pow_n;       /* [LWE lmbda] precomputed r_i^n mod N^2          */

    int      zkw_bits;
    int      zkn_digits;
    mpz_t ***zkw;           /* GMP path: [n_digits][2^w-1][LWE lmbda]         */

    /* IFMA path -- populated only by compress_m512.c.  Fields are present
     * in both builds so the struct layout is stable across paths.            */
    uint64_t  m_inv_neg;
    uint64_t  n_sq_l[MONT_N80];
    uint64_t  R2_l[MONT_N80];
    uint64_t *zkw_mont;     /* flat: [d][k][i][limb]                          */

    /* CRT half-size Mont state (40-limb).                                    */
    uint64_t  m_inv_neg_p, m_inv_neg_q;
    uint64_t  p_sq_l[MONT_N40];
    uint64_t  q_sq_l[MONT_N40];
    uint64_t  R2_p_l[MONT_N40];
    uint64_t  R2_q_l[MONT_N40];
} compress_ctx_t;

compress_ctx_t *compress_ctx_setup(int lmbda, int n_cts,
                                   int log_q, int log_p, int pl_bits);
void            compress_ctx_free(compress_ctx_t *ctx);

void  precompute_window(compress_ctx_t *ctx);
void  precompute_random(compress_ctx_t *ctx, int n_thrds);
void  compress_key     (compress_ctx_t *ctx, int n_thrds);

int   compress_response(compress_ctx_t *ctx,
                        const uint32_t *lwe_a,
                        const uint32_t *lwe_b,
                        mpz_t          *out);

void  decompress_response(compress_ctx_t *ctx,
                          mpz_t   *answer,
                          uint8_t *plaintexts,
                          int      n_thrds);

/* ============================================================================
 *  Shared helpers used by both compress.c and compress_m512.c.
 *  Static-inlined here so each TU gets its own copy (no linker collisions).
 *  paillier_decrypt is path-specific (uses mont_n80_modexp on IFMA, mpz_powm on
 *  GMP) so it's defined in each .c file rather than here.
 * ============================================================================ */

static inline int ceil_log2_u32(uint32_t x)
{
    int b = 0;
    if (x == 0) return 0;
    x--;
    while (x) { b++; x >>= 1; }
    return b;
}

static inline void mod_switch_u32(uint32_t *out, const uint32_t *in, size_t n,
                                  int log_q, int log_r)
{
    int shift = log_q - log_r;
    uint32_t round_bit = 1u << (shift - 1);
    uint32_t mask = (log_r >= 32) ? 0xffffffffu : ((1u << log_r) - 1u);
    for (size_t i = 0; i < n; i++)
        out[i] = ((in[i] + round_bit) >> shift) & mask;
}

static inline void gen_safe_prime(mpz_t p, int bits, gmp_randstate_t rs)
{
    do {
        mpz_urandomb(p, rs, bits);
        mpz_setbit(p, bits - 1);
        mpz_setbit(p, 0);
        mpz_nextprime(p, p);
    } while ((int)mpz_sizeinbase(p, 2) != bits);
}

static inline void paillier_keygen(compress_ctx_t *ctx, gmp_randstate_t rs)
{
    int half = ctx->pl_bits / 2;
    mpz_t pm1, qm1, gcd;
    mpz_inits(pm1, qm1, gcd, NULL);

    do {
        gen_safe_prime(ctx->p, half, rs);
        gen_safe_prime(ctx->q, half, rs);
        mpz_mul(ctx->n, ctx->p, ctx->q);
    } while ((int)mpz_sizeinbase(ctx->n, 2) != ctx->pl_bits);

    mpz_mul(ctx->n_sq, ctx->n, ctx->n);
    mpz_mul(ctx->p_sq, ctx->p, ctx->p);
    mpz_mul(ctx->q_sq, ctx->q, ctx->q);

    mpz_sub_ui(pm1, ctx->p, 1);
    mpz_sub_ui(qm1, ctx->q, 1);
    mpz_set(ctx->exp_p, pm1);
    mpz_set(ctx->exp_q, qm1);

    mpz_gcd(gcd, pm1, qm1);
    mpz_mul(ctx->d_xp, pm1, qm1);
    mpz_divexact(ctx->d_xp, ctx->d_xp, gcd);

    /* g = N + 1 -> mu = d_xp^(-1) mod N. */
    mpz_invert(ctx->mu, ctx->d_xp, ctx->n);

    /* CRT decryption helpers.  For x in {p, q}:
     *   h_x = L_x((N+1)^(x-1) mod x^2)^(-1) mod x  where L_x(y) = (y-1)/x mod x.
     * (N+1)^(x-1) mod x^2 = 1 + (x-1)*N mod x^2  (binomial truncation).
     * So L_x(...) = (x-1)*(N/x) mod x.                                       */
    mpz_t g, gx, lx;
    mpz_inits(g, gx, lx, NULL);
    mpz_add_ui(g, ctx->n, 1);

    mpz_powm(gx, g, ctx->exp_p, ctx->p_sq);
    mpz_sub_ui(gx, gx, 1);
    mpz_divexact(lx, gx, ctx->p);
    mpz_mod(lx, lx, ctx->p);
    mpz_invert(ctx->h_p, lx, ctx->p);

    mpz_powm(gx, g, ctx->exp_q, ctx->q_sq);
    mpz_sub_ui(gx, gx, 1);
    mpz_divexact(lx, gx, ctx->q);
    mpz_mod(lx, lx, ctx->q);
    mpz_invert(ctx->h_q, lx, ctx->q);

    mpz_invert(ctx->p_inv_q, ctx->p, ctx->q);

    mpz_clears(g, gx, lx, NULL);
    mpz_clears(pm1, qm1, gcd, NULL);
}

#endif /* COMPRESS_H */
