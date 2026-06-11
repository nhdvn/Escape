#ifndef MONT_N80_H
#define MONT_N80_H

/*
 * mont_n80.h -- Full-size Mont primitives (N=80 limbs of 52 bits = 4160
 * bits, holds N^2 for pl_bits=2048).  Used by the precompute / compress
 * hot path (Mont state stays at full size end-to-end) and by the scalar
 * paillier_decrypt fallback.
 *
 * Public API
 * ----------
 *   mont_n80_mpz_to_limbs52 / mont_n80_limbs52_to_mpz : mpz <-> limb array
 *   mont_compute_m_inv_neg : (-m)^(-1) mod 2^52  (size-agnostic; only uses m[0])
 *   mont_n80_modmul        : scalar CIOS Mont modmul (1 ct, register-resident T)
 *   mont_n80_modexp        : 4-bit windowed left-to-right square-and-multiply
 *   mont_n80_w16_modmul    : 16-way Mont modmul (16 cts in 2*8 lanes)
 *                            ~6.5x GMP per ct on Ice Lake-SP
 *   mont_n80_w16_modexp    : 16-way Mont modexp (shared exponent, 16 bases)
 *
 * Internal helpers (also reused by mont_n40.h):
 *   mont_w16_mul_n{20,40,80}_x2 : Karatsuba levels via MONT_W16_KARA_X2_BODY.
 *   mont_w16_mul_NxN_x2         : schoolbook leaf, 2-deep pipelined.
 *
 * See mont.h for shared macro families (V_*, M16_*, M16N_*, KARA_X2_BODY)
 * and pipelining / Karatsuba algorithmic details.
 */

#include "mont.h"

#include <string.h>
#include <gmp.h>

static inline void mont_n80_mpz_to_limbs52(uint64_t out[MONT_N80], const mpz_t z)
{
    mpz_t t; mpz_init_set(t, z);
    for (int i = 0; i < MONT_N80; i++) {
        out[i] = mpz_get_ui(t) & MONT_LIMB_MASK;
        mpz_fdiv_q_2exp(t, t, MONT_LIMB_BITS);
    }
    mpz_clear(t);
}

static inline void mont_n80_limbs52_to_mpz(mpz_t out, const uint64_t in[MONT_N80])
{
    mpz_set_ui(out, 0);
    for (int i = MONT_N80 - 1; i >= 0; i--) {
        mpz_mul_2exp(out, out, MONT_LIMB_BITS);
        mpz_add_ui(out, out, in[i] & MONT_LIMB_MASK);
    }
}

static inline uint64_t mont_compute_m_inv_neg(const uint64_t m[MONT_N80])
{
    uint64_t x = m[0] & MONT_LIMB_MASK, y = x;
    for (int i = 0; i < 5; i++) y = y * (2 - x * y);
    return ((uint64_t)(-y)) & MONT_LIMB_MASK;
}

__attribute__((always_inline))
static inline void mont_n80_modmul(uint64_t c[MONT_N80], const uint64_t a[MONT_N80],
            const uint64_t b[MONT_N80], const uint64_t m[MONT_N80], uint64_t m_inv_neg)
{
    enum { NT = MONT_N80 / 8 + 3 };          /* 11 zmm "rows" of 8 limbs */
    __m512i T[NT];
    for (int k = 0; k < NT; k++) T[k] = V_ZERO;

    for (int i = 0; i < MONT_N80; i++) {
        __m512i a_v = V_SET1(a[i]);
        __m512i hi_m, hi_r, hi_sh, comb;
        __m512i b0 = V_LOAD(&b[0]);
        __m512i mul_hi_0 = V_HI(V_ZERO, a_v, b0);
        T[0] = V_LO(T[0], a_v, b0);

        uint64_t T0_lane0 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(T[0]));
        uint64_t u = ((T0_lane0 & MONT_LIMB_MASK) * m_inv_neg) & MONT_LIMB_MASK;
        __m512i u_v = V_SET1(u);

        __m512i m0 = V_LOAD(&m[0]);
        __m512i redc_hi_0 = V_HI(V_ZERO, u_v, m0);
        T[0] = V_LO(T[0], u_v, m0);
        __m512i prev_comb = V_ADD(mul_hi_0, redc_hi_0);
        hi_sh = V_ALIGN(prev_comb, V_ZERO, 7);
        T[0] = V_ADD(T[0], hi_sh);

        #define MONT_J(idx) ({                     \
            __m512i b_v = V_LOAD(&b[(idx)*8]);     \
            __m512i m_v = V_LOAD(&m[(idx)*8]);     \
            hi_m  = V_HI(V_ZERO, a_v, b_v);        \
            T[idx] = V_LO(T[idx], a_v, b_v);       \
            hi_r  = V_HI(V_ZERO, u_v, m_v);        \
            T[idx] = V_LO(T[idx], u_v, m_v);       \
            comb  = V_ADD(hi_m, hi_r);             \
            hi_sh = V_ALIGN(comb, prev_comb, 7);   \
            T[idx] = V_ADD(T[idx], hi_sh);         \
            prev_comb = comb; })
        MONT_J(1); MONT_J(2); MONT_J(3); MONT_J(4); MONT_J(5);
        MONT_J(6); MONT_J(7); MONT_J(8); MONT_J(9);
        #undef MONT_J

        T[10] = V_ADD(T[10], V_ALIGN(V_ZERO, comb, 7));

        T0_lane0 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(T[0]));
        T[0] = V_ADD(T[0], _mm512_maskz_set1_epi64(0x02,
                          (long long)(T0_lane0 >> MONT_LIMB_BITS)));

        for (int k = 0; k < NT - 1; k++) T[k] = V_ALIGN(T[k+1], T[k], 1);
        T[NT-1] = V_ALIGN(V_ZERO, T[NT-1], 1);
    }

    uint64_t T_arr[NT * 8] __attribute__((aligned(64)));
    for (int k = 0; k < NT; k++)
        _mm512_storeu_si512((void *)&T_arr[k*8], T[k]);

    uint64_t carry = 0;
    for (int k = 0; k < NT * 8; k++) {
        uint64_t s_v = T_arr[k] + carry;
        T_arr[k] = s_v & MONT_LIMB_MASK;
        carry = s_v >> MONT_LIMB_BITS;
    }
    uint64_t s_buf[MONT_N80];
    int64_t borrow = 0;
    for (int k = 0; k < MONT_N80; k++) {
        int64_t d = (int64_t)T_arr[k] - (int64_t)m[k] - borrow;
        if (d < 0) { s_buf[k] = (uint64_t)(d + (1LL<<MONT_LIMB_BITS)); borrow = 1; }
        else       { s_buf[k] = (uint64_t)d; borrow = 0; }
    }
    if (borrow == 0) for (int k = 0; k < MONT_N80; k++) c[k] = s_buf[k];
    else             for (int k = 0; k < MONT_N80; k++) c[k] = T_arr[k];
}

static inline void mont_n80_modexp(uint64_t out_l[MONT_N80], const uint64_t inp_l[MONT_N80],
                               const mpz_t exp, const uint64_t m_l[MONT_N80],
                               const uint64_t R2_l[MONT_N80], uint64_t m_inv_neg)
{
    if (mpz_sgn(exp) == 0) {
        out_l[0] = 1;
        for (int k = 1; k < MONT_N80; k++) out_l[k] = 0;
        return;
    }

    uint64_t one_l[MONT_N80] = { 1 };
    uint64_t pow_M[16][MONT_N80], res[MONT_N80];
    mont_n80_modmul(pow_M[0], one_l,  R2_l, m_l, m_inv_neg);
    mont_n80_modmul(pow_M[1], inp_l, R2_l, m_l, m_inv_neg);
    for (int k = 2; k < 16; k++)
        mont_n80_modmul(pow_M[k], pow_M[k-1], pow_M[1], m_l, m_inv_neg);
    memcpy(res, pow_M[0], sizeof(res));

    mp_bitcnt_t n_bits = mpz_sizeinbase(exp, 2);
    int top = (int)(n_bits % 4); if (top == 0) top = 4;
    mp_bitcnt_t bit = n_bits;

    unsigned w0 = 0;
    for (int j = top - 1; j >= 0; j--) { bit--; w0 = (w0<<1) | (unsigned)mpz_tstbit(exp, bit); }
    if (w0 != 0) mont_n80_modmul(res, res, pow_M[w0], m_l, m_inv_neg);

    while (bit >= 4) {
        for (int j = 0; j < 4; j++) mont_n80_modmul(res, res, res, m_l, m_inv_neg);
        bit -= 4;
        unsigned w = ((unsigned)mpz_tstbit(exp, bit+3) << 3)
                   | ((unsigned)mpz_tstbit(exp, bit+2) << 2)
                   | ((unsigned)mpz_tstbit(exp, bit+1) << 1)
                   | ((unsigned)mpz_tstbit(exp, bit+0));
        if (w != 0) mont_n80_modmul(res, res, pow_M[w], m_l, m_inv_neg);
    }
    mont_n80_modmul(out_l, res, one_l, m_l, m_inv_neg);
}

static __attribute__((always_inline)) inline
void mont_n80_w16_modmul(__m512i *out_a, __m512i *out_b, const __m512i *a_a, const __m512i *b_a,
                const __m512i *a_b, const __m512i *b_b, const uint64_t *m_h, uint64_t m_inv_neg)
{
    enum { N = MONT_N80 };
    __m512i T_a[2*N + 1], T_b[2*N + 1];

    mont_w16_mul_n80_x2(T_a, T_b, a_a, b_a, a_b, b_b);
    T_a[2*N] = V_ZERO; T_b[2*N] = V_ZERO;

    __m512i m_inv_neg_v = V_SET1(m_inv_neg);

    for (int i = 0; i + 3 < N; i += 4) {
        if (i > 0) {
            T_a[i] = V_ADD(T_a[i], V_SRAI(T_a[i-1], 52));
            T_b[i] = V_ADD(T_b[i], V_SRAI(T_b[i-1], 52));
        }
        __m512i u_a_0 = V_AND(_mm512_mullo_epi64(T_a[i], m_inv_neg_v), V_MASK52);
        __m512i u_b_0 = V_AND(_mm512_mullo_epi64(T_b[i], m_inv_neg_v), V_MASK52);
        __m512i tcur_a_0 = T_a[i], tcur_b_0 = T_b[i];
        __m512i u_a_1, u_b_1, tcur_a_1, tcur_b_1;
        __m512i u_a_2, u_b_2, tcur_a_2, tcur_b_2;
        __m512i u_a_3, u_b_3, tcur_a_3, tcur_b_3;

        M16_STEP4(0, 0, 0, 0, 1); M16_STEP4(1, 0, 0, 0, 1); M16_BRING(1);
        M16_STEP4(2, 0, 0, 0, 2); M16_STEP4(3, 1, 0, 0, 2); M16_BRING(2);
        M16_STEP4(4, 2, 0, 0, 3); M16_STEP4(5, 3, 1, 0, 3); M16_BRING(3);

        for (int k = 6; k < N; k++) M16_STEP4(k, k-2, k-4, k-6, 4);
        T_a[i+N] = tcur_a_0; T_b[i+N] = tcur_b_0;

        M16_STEP3(N-2, N-4, N-6); M16_STEP3(N-1, N-3, N-5);
        T_a[i+1+N] = tcur_a_1; T_b[i+1+N] = tcur_b_1;

        M16_STEP2(N-2, N-4); M16_STEP2(N-1, N-3);
        T_a[i+2+N] = tcur_a_2; T_b[i+2+N] = tcur_b_2;

        M16_STEP1(N-2); M16_STEP1(N-1);
        T_a[i+3+N] = tcur_a_3; T_b[i+3+N] = tcur_b_3;
    }

    for (int i = (N & ~3); i < N; i++) {
        if (i > 0) {
            T_a[i] = V_ADD(T_a[i], V_SRAI(T_a[i-1], 52));
            T_b[i] = V_ADD(T_b[i], V_SRAI(T_b[i-1], 52));
        }
        __m512i u_a = V_AND(_mm512_mullo_epi64(T_a[i], m_inv_neg_v), V_MASK52);
        __m512i u_b = V_AND(_mm512_mullo_epi64(T_b[i], m_inv_neg_v), V_MASK52);
        __m512i tcur_a = T_a[i], tcur_b = T_b[i];
        for (int j = 0; j < N; j++) {
            __m512i m_v_j = V_SET1(m_h[j]);
            tcur_a = V_LO(tcur_a, u_a, m_v_j);
            tcur_b = V_LO(tcur_b, u_b, m_v_j);
            __m512i tnext_a = V_HI(T_a[i+j+1], u_a, m_v_j);
            __m512i tnext_b = V_HI(T_b[i+j+1], u_b, m_v_j);
            T_a[i+j] = tcur_a; T_b[i+j] = tcur_b;
            tcur_a = tnext_a;  tcur_b = tnext_b;
        }
        T_a[i+N] = tcur_a; T_b[i+N] = tcur_b;
    }

    for (int k = 0; k < 2*N; k++) {
        __m512i ca = V_SRAI(T_a[k], 52), cb = V_SRAI(T_b[k], 52);
        T_a[k]   = V_AND(T_a[k], V_MASK52); T_b[k]   = V_AND(T_b[k], V_MASK52);
        T_a[k+1] = V_ADD(T_a[k+1], ca);     T_b[k+1] = V_ADD(T_b[k+1], cb);
    }

    __m512i sub_a[N], sub_b[N];
    __m512i borrow_a = V_ZERO, borrow_b = V_ZERO;
    for (int k = 0; k < N; k++) {
        __m512i m_v_k = V_SET1(m_h[k]);
        __m512i da = V_SUB(V_SUB(T_a[N+k], m_v_k), borrow_a);
        __m512i db = V_SUB(V_SUB(T_b[N+k], m_v_k), borrow_b);
        sub_a[k] = V_AND(da, V_MASK52); sub_b[k] = V_AND(db, V_MASK52);
        borrow_a = V_SRLI(da, 63);      borrow_b = V_SRLI(db, 63);
    }
    __mmask8 ua = _mm512_cmpeq_epi64_mask(borrow_a, V_ZERO);
    __mmask8 ub = _mm512_cmpeq_epi64_mask(borrow_b, V_ZERO);
    for (int k = 0; k < N; k++) {
        out_a[k] = _mm512_mask_blend_epi64(ua, T_a[N+k], sub_a[k]);
        out_b[k] = _mm512_mask_blend_epi64(ub, T_b[N+k], sub_b[k]);
    }
}

/* 16-way modexp: 16 cts share the same modulus + exponent, only base differs.
 * 4-bit windowed left-to-right square-and-multiply, identical to mont_n80_modexp 
 * but use mont_n80_w16_modmul. Input inp_a/_b and Output out_a/_b in natural form. */
static inline void mont_n80_w16_modexp(__m512i out_a[MONT_N80], __m512i out_b[MONT_N80],
            const __m512i inp_a[MONT_N80], const __m512i inp_b[MONT_N80], const mpz_t exp, 
            const uint64_t m_l[MONT_N80], const uint64_t R2_l[MONT_N80], uint64_t m_inv_neg)
{
    /* Broadcast R2 and 1 to vector form (same value across all 8 lanes). */
    __m512i R2_v[MONT_N80], one_v[MONT_N80];
    for (int i = 0; i < MONT_N80; i++) {
        R2_v[i]  = _mm512_set1_epi64((long long)R2_l[i]);
        one_v[i] = _mm512_setzero_si512();
    }
    one_v[0] = _mm512_set1_epi64(1);

    if (mpz_sgn(exp) == 0) {
        for (int k = 0; k < MONT_N80; k++) {
            out_a[k] = one_v[k];
            out_b[k] = one_v[k];
        }
        return;
    }

    /* pow_M[k] = Mont(base^k); pow_M[0] = Mont(1) = R, pow_M[1] = Mont(base).
     * mont_n80_w16_modmul args: (out_a, out_b, factor1_a, factor2_a, factor1_b, factor2_b). */
    __m512i pow_M_a[16][MONT_N80], pow_M_b[16][MONT_N80];
    mont_n80_w16_modmul(pow_M_a[0], pow_M_b[0], one_v,  R2_v, one_v,  R2_v, m_l, m_inv_neg);
    mont_n80_w16_modmul(pow_M_a[1], pow_M_b[1], inp_a, R2_v, inp_b, R2_v, m_l, m_inv_neg);
    
    for (int k = 2; k < 16; k++)
        mont_n80_w16_modmul(pow_M_a[k], pow_M_b[k], pow_M_a[k-1], pow_M_a[1],
                            pow_M_b[k-1], pow_M_b[1], m_l, m_inv_neg);

    __m512i res_a[MONT_N80], res_b[MONT_N80];
    memcpy(res_a, pow_M_a[0], sizeof(res_a));
    memcpy(res_b, pow_M_b[0], sizeof(res_b));

    mp_bitcnt_t n_bits = mpz_sizeinbase(exp, 2);
    int top = (int)(n_bits % 4); if (top == 0) top = 4;
    mp_bitcnt_t bit = n_bits;

    unsigned w0 = 0;
    for (int j = top - 1; j >= 0; j--) {
        bit--;
        w0 = (w0 << 1) | (unsigned)mpz_tstbit(exp, bit);
    }
    if (w0 != 0) {
        memcpy(res_a, pow_M_a[w0], sizeof(res_a));
        memcpy(res_b, pow_M_b[w0], sizeof(res_b));
    }

    while (bit >= 4) {
        for (int j = 0; j < 4; j++)
            mont_n80_w16_modmul(res_a, res_b, res_a, res_a,
                                res_b, res_b, m_l, m_inv_neg);
        bit -= 4;
        unsigned w = ((unsigned)mpz_tstbit(exp, bit+3) << 3)
                   | ((unsigned)mpz_tstbit(exp, bit+2) << 2)
                   | ((unsigned)mpz_tstbit(exp, bit+1) << 1)
                   | ((unsigned)mpz_tstbit(exp, bit+0));
        if (w != 0)
            mont_n80_w16_modmul(res_a, res_b, res_a, pow_M_a[w],
                                res_b, pow_M_b[w], m_l, m_inv_neg);
    }

    /* Mont -> natural: out = res * 1 / R. */
    mont_n80_w16_modmul(out_a, out_b, res_a, one_v, res_b, one_v, m_l, m_inv_neg);
}

#endif /* MONT_N80_H */
