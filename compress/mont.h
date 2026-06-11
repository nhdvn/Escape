#ifndef MONT_H
#define MONT_H

/* ============================================================================
 *  mont.h -- AVX-512 IFMA52 Montgomery primitives, shared building blocks.
 *
 *  File hierarchy
 *  --------------
 *    mont.h     : sizing constants, V_* intrinsic shortcuts, and macro
 *                 families (M16_*, M16N_*, MONT_W16_KARA_X2_BODY) shared
 *                 across both modulus sizes.
 *    mont_n80.h : full-size primitives (N=80 limbs = 4160 bits, holds N^2
 *                 for pl_bits=2048).  Includes mont.h.
 *    mont_n40.h : half-size primitives (N=40 limbs = 2080 bits, holds p^2
 *                 for the Paillier-CRT path).  Includes mont_n80.h
 *                 (transitively brings mont.h and the Karatsuba helpers).
 *
 *  Naming convention (functions defined in mont_n80.h / mont_n40.h)
 *  ---------------------------------------------------------------
 *    mont_nN_*    : N-limb scalar primitive    (one ciphertext per call).
 *    mont_nN_w16_*: N-limb 16-way SIMD primitive (16 cts via 2 instances
 *                                                 of 8 vector lanes).
 *    mont_w16_mul_nN_x2 / NxN_x2 : Karatsuba helpers shared by w16 paths.
 *    mont_compute_m_inv_neg : size-agnostic (only reads m[0]).
 *
 *  16-way internals (used by mont_nN_w16_modmul at both sizes)
 *  -----------------------------------------------------------
 *    Phase 1: 3-level Karatsuba (N -> N/2 -> N/4 -> N/8 schoolbook leaf).
 *             Output is signed-lazy 2N-limb T.
 *             mont_w16_mul_n80_x2 / n40 / n20 (kara levels) and
 *             mont_w16_mul_NxN_x2 (schoolbook leaf, 2-deep pipelined).
 *    Phase 2: Mont CIOS reduce, 4-deep software-pipelined j-loop.
 *    Final  : srai canonicalize (kara is signed-lazy through Mont reduce),
 *             then conditional subtract m -> result in [0, m).
 *
 *  Pipelining
 *  ----------
 *    Both phases run multiple "iters" (outer-loop indices) concurrently:
 *    iter X+1 lags iter X by 2 j-steps; iter X+1's HI register-forwards
 *    iter X's just-computed tcur (no memory roundtrip).  Phase 1's NxN
 *    schoolbook is 2-deep (2 iters in flight), Phase 2's reduce is 4-deep.
 *    16 chains in flight in Phase 2 saturate vpmadd52 ports + hide spill
 *    latency.
 *
 *  Macro families (file-scope; expand inside their callers' local scope)
 *  --------------
 *    V_*    : __m512i intrinsic shortcuts (V_LO, V_HI, V_ADD, V_SET1, ...).
 *    M16N_* : building blocks for mont_w16_mul_NxN_x2 (schoolbook NxN);
 *             refer to locals tcur_a/b_X, av_a/b_X, c_a/b, b_a/b, i.
 *             SOLO = 1-iter step, PAIR = 2-iter paired step.
 *    M16_*  : building blocks for *_w16_modmul Phase-2 reduce; refer to
 *             T_a/b, m_h, m_inv_neg_v, i, plus per-iter tcur_a/b_X,
 *             u_a/b_X, tn_a/b_X.  STEPK = K-iter paired step.  BRING(X) =
 *             carry-prop T[i+X] and seed iter X's u/tcur.
 *    MONT_W16_KARA_X2_BODY : Karatsuba splitter body, parameterized by
 *             half-size and inner z0/z2 multiply -- used by all three
 *             kara levels (mont_w16_mul_n{20,40,80}_x2).
 *    All multi-statement macros use GNU statement expressions ({ ... }).
 *
 *  Pair-iteration pattern (mont_w16_mul_NxN_x2)
 *  --------------------------------------------
 *    pre-roll  iter 0 j=0, 1 (alone)         [SOLO]
 *    bring     iter 1 with tcur_1 = c[i+1]   [register init]
 *    main      iter 0 j=2..bsz-1, iter 1 j=0..bsz-3 (paired, lag 2) [PAIR]
 *    post-roll iter 1 j=bsz-2, bsz-1 (alone) [SOLO]
 *
 *  4-tuple pattern (*_w16_modmul Phase 2, step k: iter X at j=k-2*X)
 *  -----------------------------------------------------------------
 *    carry-prop T[i] from T[i-1] (skip i=0); seed iter 0's u/tcur
 *    pre-roll  3 stages bringing iters 1, 2, 3 online (2 STEP4s each)
 *    steady    all 4 iters in flight, k=6..N-1                 [STEP4]
 *    post-roll drain iter 1 (2 STEP3s), iter 2 (2 STEP2s), iter 3 (2 STEP1s)
 * ============================================================================ */

#include <stdint.h>
#include <immintrin.h>

#define MONT_N80    80
#define MONT_LIMB_BITS 52
#define MONT_LIMB_MASK ((1ULL << MONT_LIMB_BITS) - 1)

/* ---- Vector intrinsic shortcuts ---------------------------------------- */

#define V_ZERO         _mm512_setzero_si512()
#define V_SET1(x)      _mm512_set1_epi64((long long)(x))
#define V_LOAD(p)      _mm512_loadu_si512((const void *)(p))
#define V_ADD(a,b)     _mm512_add_epi64((a),(b))
#define V_SUB(a,b)     _mm512_sub_epi64((a),(b))
#define V_AND(a,b)     _mm512_and_si512((a),(b))
#define V_SRLI(a,n)    _mm512_srli_epi64((a),(n))
#define V_SRAI(a,n)    _mm512_srai_epi64((a),(n))
#define V_LO(t,a,b)    _mm512_madd52lo_epu64((t),(a),(b))
#define V_HI(t,a,b)    _mm512_madd52hi_epu64((t),(a),(b))
#define V_ALIGN(h,l,n) _mm512_alignr_epi64((h),(l),(n))
#define V_MASK52       V_SET1(MONT_LIMB_MASK)

/* ---- 16-way Phase-2 reduce helper macros -------------------------------- */

#define M16_BRING(X)                                                     \
    ({ __m512i tx_a = V_ADD(T_a[i+(X)], V_SRAI(T_a[i+(X)-1], 52));       \
       __m512i tx_b = V_ADD(T_b[i+(X)], V_SRAI(T_b[i+(X)-1], 52));       \
       u_a_##X = V_AND(_mm512_mullo_epi64(tx_a, m_inv_neg_v), V_MASK52); \
       u_b_##X = V_AND(_mm512_mullo_epi64(tx_b, m_inv_neg_v), V_MASK52); \
       tcur_a_##X = tx_a; tcur_b_##X = tx_b; })

#define M16_LO(X, M)                                        \
    ({ tcur_a_##X = V_LO(tcur_a_##X, u_a_##X, M);           \
       tcur_b_##X = V_LO(tcur_b_##X, u_b_##X, M); })        \

#define M16_HI_FWD(X, prev, M)                              \
    ({ tn_a_##X = V_HI(tcur_a_##prev, u_a_##X, M);          \
       tn_b_##X = V_HI(tcur_b_##prev, u_b_##X, M); })       \

#define M16_HI_MEM(X, ADDR, M)                              \
    ({ tn_a_##X = V_HI(T_a[ADDR], u_a_##X, M);              \
       tn_b_##X = V_HI(T_b[ADDR], u_b_##X, M); })           \

#define M16_STORE(X, ADDR)                                  \
    ({ T_a[ADDR] = tcur_a_##X; T_b[ADDR] = tcur_b_##X; })   \

#define M16_ADV(X)                                          \
    ({ tcur_a_##X = tn_a_##X; tcur_b_##X = tn_b_##X; })     \

#define M16_STEP4(j0, j1, j2, j3, nactive)                            \
    ({ __m512i m0_ = V_SET1(m_h[(j0)]);                               \
       __m512i m1_ = V_SET1(m_h[(j1)]);                               \
       __m512i m2_ = V_SET1(m_h[(j2)]);                               \
       __m512i m3_ = V_SET1(m_h[(j3)]);                               \
       __m512i tn_a_0, tn_b_0, tn_a_1, tn_b_1;                        \
       __m512i tn_a_2, tn_b_2, tn_a_3, tn_b_3;                        \
       M16_LO(0, m0_); M16_HI_MEM(0, i+(j0)+1, m0_);                  \
       if ((nactive) >= 2) { M16_LO(1, m1_); M16_HI_FWD(1, 0, m1_); } \
       if ((nactive) >= 3) { M16_LO(2, m2_); M16_HI_FWD(2, 1, m2_); } \
       if ((nactive) >= 4) { M16_LO(3, m3_); M16_HI_FWD(3, 2, m3_); } \
       if ((nactive) >= 4) M16_STORE(3, i+3+(j3));                    \
       if ((nactive) >= 3) M16_STORE(2, i+2+(j2));                    \
       if ((nactive) >= 2) M16_STORE(1, i+1+(j1));                    \
       M16_STORE(0, i+(j0));                                          \
       M16_ADV(0);                                                    \
       if ((nactive) >= 2) M16_ADV(1);                                \
       if ((nactive) >= 3) M16_ADV(2);                                \
       if ((nactive) >= 4) M16_ADV(3); })

#define M16_STEP3(j1, j2, j3)                                         \
    ({ __m512i m1_ = V_SET1(m_h[(j1)]);                               \
       __m512i m2_ = V_SET1(m_h[(j2)]);                               \
       __m512i m3_ = V_SET1(m_h[(j3)]);                               \
       __m512i tn_a_1, tn_b_1, tn_a_2, tn_b_2, tn_a_3, tn_b_3;        \
       M16_LO(1, m1_); M16_HI_MEM(1, i+1+(j1)+1, m1_);                \
       M16_LO(2, m2_); M16_HI_FWD(2, 1, m2_);                         \
       M16_LO(3, m3_); M16_HI_FWD(3, 2, m3_);                         \
       M16_STORE(3, i+3+(j3));                                        \
       M16_STORE(2, i+2+(j2));                                        \
       M16_STORE(1, i+1+(j1));                                        \
       M16_ADV(1); M16_ADV(2); M16_ADV(3); })

#define M16_STEP2(j2, j3)                                             \
    ({ __m512i m2_ = V_SET1(m_h[(j2)]);                               \
       __m512i m3_ = V_SET1(m_h[(j3)]);                               \
       __m512i tn_a_2, tn_b_2, tn_a_3, tn_b_3;                        \
       M16_LO(2, m2_); M16_HI_MEM(2, i+2+(j2)+1, m2_);                \
       M16_LO(3, m3_); M16_HI_FWD(3, 2, m3_);                         \
       M16_STORE(3, i+3+(j3));                                        \
       M16_STORE(2, i+2+(j2));                                        \
       M16_ADV(2); M16_ADV(3); })

#define M16_STEP1(j3)                                                 \
    ({ __m512i m3_ = V_SET1(m_h[(j3)]);                               \
       __m512i tn_a_3, tn_b_3;                                        \
       M16_LO(3, m3_); M16_HI_MEM(3, i+3+(j3)+1, m3_);                \
       M16_STORE(3, i+3+(j3));                                        \
       M16_ADV(3); })

/* ---- 16-way schoolbook NxN helper macros (used by mont_w16_mul_NxN_x2)
 *      Reference local-scope c_a, c_b, b_a, b_b, av_a/b_X, tcur_a/b_X,
 *      tn_a/b_X, i.  SOLO = 1-iter step, PAIR = 2-iter paired step. -------- */

#define M16N_LO(X, BV_A, BV_B)                                   \
    ({ tcur_a_##X = V_LO(tcur_a_##X, av_a_##X, BV_A);            \
       tcur_b_##X = V_LO(tcur_b_##X, av_b_##X, BV_B); })         \

#define M16N_HI_FWD(X, prev, BV_A, BV_B)                         \
    ({ tn_a_##X = V_HI(tcur_a_##prev, av_a_##X, BV_A);           \
       tn_b_##X = V_HI(tcur_b_##prev, av_b_##X, BV_B); })        \

#define M16N_HI_MEM(X, ADDR, BV_A, BV_B)                         \
    ({ tn_a_##X = V_HI(c_a[ADDR], av_a_##X, BV_A);               \
       tn_b_##X = V_HI(c_b[ADDR], av_b_##X, BV_B); })            \

#define M16N_STORE(X, ADDR)                                      \
    ({ c_a[ADDR] = tcur_a_##X; c_b[ADDR] = tcur_b_##X; })        \

#define M16N_ADV(X)                                              \
    ({ tcur_a_##X = tn_a_##X; tcur_b_##X = tn_b_##X; })          \

#define M16N_SOLO(X, j_, ADDR_STORE)                             \
    ({ __m512i bv_a = b_a[(j_)], bv_b = b_b[(j_)];               \
       __m512i tn_a_##X, tn_b_##X;                               \
       M16N_LO(X, bv_a, bv_b);                                   \
       M16N_HI_MEM(X, (ADDR_STORE)+1, bv_a, bv_b);               \
       M16N_STORE(X, ADDR_STORE);                                \
       M16N_ADV(X); })

#define M16N_PAIR(j_)                                            \
    ({ __m512i bv_a_0 = b_a[(j_)],   bv_b_0 = b_b[(j_)];         \
       __m512i bv_a_1 = b_a[(j_)-2], bv_b_1 = b_b[(j_)-2];       \
       __m512i tn_a_0, tn_b_0, tn_a_1, tn_b_1;                   \
       M16N_LO(0, bv_a_0, bv_b_0);                               \
       M16N_HI_MEM(0, i+(j_)+1, bv_a_0, bv_b_0);                 \
       M16N_LO(1, bv_a_1, bv_b_1);                               \
       M16N_HI_FWD(1, 0, bv_a_1, bv_b_1);                        \
       M16N_STORE(1, i+(j_)-1);                                  \
       M16N_STORE(0, i+(j_));                                    \
       M16N_ADV(0); M16N_ADV(1); })

/* ---- 16-way Karatsuba splitter body (used by mont_w16_mul_n{20,40,80}_x2)
 *      SZ_HALF = N/2; MUL_INNER_z0z2 = recursion to next-level karatsuba
 *      that fills z0 (low halves) and z2 (high halves).  References local
 *      a_a, b_a, a_b, b_b (input halves), c_a, c_b (output 2N), and uses
 *      mont_w16_mul_NxN_x2 for the mid-product (sum-of-halves * sum). ----- */

#define MONT_W16_KARA_X2_BODY(SZ_HALF, MUL_INNER_z0z2)                                      \
    enum { NH = (SZ_HALF) };                                                                \
    __m512i mask52 = V_MASK52;                                                              \
    __m512i aS_a[NH+1], bS_a[NH+1], aS_b[NH+1], bS_b[NH+1];                                 \
    {                                                                                       \
        __m512i ca_a=V_ZERO, cb_a=V_ZERO, ca_b=V_ZERO, cb_b=V_ZERO;                         \
        for (int k = 0; k < NH; k++) {                                                      \
            __m512i sa_a = V_ADD(V_ADD(a_a[k], a_a[k+NH]), ca_a);                           \
            __m512i sb_a = V_ADD(V_ADD(b_a[k], b_a[k+NH]), cb_a);                           \
            __m512i sa_b = V_ADD(V_ADD(a_b[k], a_b[k+NH]), ca_b);                           \
            __m512i sb_b = V_ADD(V_ADD(b_b[k], b_b[k+NH]), cb_b);                           \
            aS_a[k] = V_AND(sa_a, mask52); bS_a[k] = V_AND(sb_a, mask52);                   \
            aS_b[k] = V_AND(sa_b, mask52); bS_b[k] = V_AND(sb_b, mask52);                   \
            ca_a = V_SRLI(sa_a, 52); cb_a = V_SRLI(sb_a, 52);                               \
            ca_b = V_SRLI(sa_b, 52); cb_b = V_SRLI(sb_b, 52);                               \
        }                                                                                   \
        aS_a[NH]=ca_a; bS_a[NH]=cb_a; aS_b[NH]=ca_b; bS_b[NH]=cb_b;                         \
    }                                                                                       \
    __m512i z0_a[2*NH], z2_a[2*NH], z_S_a[2*(NH+1)];                                        \
    __m512i z0_b[2*NH], z2_b[2*NH], z_S_b[2*(NH+1)];                                        \
    MUL_INNER_z0z2;                                                                         \
    mont_w16_mul_NxN_x2(z_S_a, z_S_b, 2*(NH+1),                                             \
                      aS_a, bS_a, aS_b, bS_b, NH+1, NH+1);                                  \
    for (int k = 0; k < 4*NH; k++) {                                                        \
        __m512i v_a = V_ZERO, v_b = V_ZERO;                                                 \
        if (k < NH) { v_a = V_ADD(v_a, z0_a[k]); v_b = V_ADD(v_b, z0_b[k]); }               \
        if (k >= 2*NH) { v_a = V_ADD(v_a, z2_a[k-2*NH]); v_b = V_ADD(v_b, z2_b[k-2*NH]); }  \
        if (k >= NH && k < 2*NH) { v_a = V_ADD(v_a, z0_a[k]); v_b = V_ADD(v_b, z0_b[k]); }  \
        if (k >= NH && k < NH + 2*(NH+1)) {                                                 \
            v_a = V_ADD(v_a, z_S_a[k-NH]); v_b = V_ADD(v_b, z_S_b[k-NH]);                   \
        }                                                                                   \
        if (k >= NH && k < NH + 2*NH) {                                                     \
            v_a = V_SUB(v_a, z0_a[k-NH]); v_b = V_SUB(v_b, z0_b[k-NH]);                     \
            v_a = V_SUB(v_a, z2_a[k-NH]); v_b = V_SUB(v_b, z2_b[k-NH]);                     \
        }                                                                                   \
        c_a[k] = v_a; c_b[k] = v_b;                                                         \
    }

/* ---- Karatsuba helpers (size-parametric, used by both mont_n80 and mont_n40)
 *      mont_w16_mul_NxN_x2 : schoolbook leaf, 2-deep pipelined.
 *      mont_w16_mul_n/20/40/80_x2 : Karatsuba levels via KARA_X2_BODY. ---- */

static __attribute__((always_inline)) inline
void mont_w16_mul_NxN_x2(__m512i *c_a, __m512i *c_b, int csz, const __m512i *a_a,
    const __m512i *b_a, const __m512i *a_b, const __m512i *b_b, int asz, int bsz)
{
    for (int k = 0; k < csz; k++) { c_a[k] = V_ZERO; c_b[k] = V_ZERO; }

    if (bsz < 4) {
        for (int i = 0; i < asz; i++) {
            __m512i av_a = a_a[i], av_b = a_b[i];
            __m512i tcur_a = c_a[i], tcur_b = c_b[i];
            for (int j = 0; j < bsz; j++) {
                __m512i bv_a = b_a[j], bv_b = b_b[j];
                tcur_a = V_LO(tcur_a, av_a, bv_a);
                tcur_b = V_LO(tcur_b, av_b, bv_b);
                __m512i tnext_a = V_HI(c_a[i+j+1], av_a, bv_a);
                __m512i tnext_b = V_HI(c_b[i+j+1], av_b, bv_b);
                c_a[i+j] = tcur_a; c_b[i+j] = tcur_b;
                tcur_a = tnext_a;  tcur_b = tnext_b;
            }
            c_a[i+bsz] = tcur_a; c_b[i+bsz] = tcur_b;
        }
        return;
    }

    int i = 0;
    for (; i + 1 < asz; i += 2) {
        __m512i av_a_0 = a_a[i],   av_b_0 = a_b[i];
        __m512i av_a_1 = a_a[i+1], av_b_1 = a_b[i+1];
        __m512i tcur_a_0 = c_a[i], tcur_b_0 = c_b[i];
        M16N_SOLO(0, 0, i);
        M16N_SOLO(0, 1, i+1);
        __m512i tcur_a_1 = c_a[i+1], tcur_b_1 = c_b[i+1];

        for (int j = 2; j < bsz; j++) M16N_PAIR(j);
        c_a[i+bsz] = tcur_a_0; c_b[i+bsz] = tcur_b_0;

        M16N_SOLO(1, bsz-2, i+bsz-1);
        M16N_SOLO(1, bsz-1, i+bsz);
        c_a[i+1+bsz] = tcur_a_1; c_b[i+1+bsz] = tcur_b_1;
    }

    if (i < asz) {
        __m512i av_a = a_a[i], av_b = a_b[i];
        __m512i tcur_a = c_a[i], tcur_b = c_b[i];
        for (int j = 0; j < bsz; j++) {
            __m512i bv_a = b_a[j], bv_b = b_b[j];
            tcur_a = V_LO(tcur_a, av_a, bv_a);
            tcur_b = V_LO(tcur_b, av_b, bv_b);
            __m512i tnext_a = V_HI(c_a[i+j+1], av_a, bv_a);
            __m512i tnext_b = V_HI(c_b[i+j+1], av_b, bv_b);
            c_a[i+j] = tcur_a; c_b[i+j] = tcur_b;
            tcur_a = tnext_a;  tcur_b = tnext_b;
        }
        c_a[i+bsz] = tcur_a; c_b[i+bsz] = tcur_b;
    }
}

static __attribute__((always_inline)) inline
void mont_w16_mul_n20_x2(__m512i c_a[40], __m512i c_b[40],
                           const __m512i a_a[20], const __m512i b_a[20],
                           const __m512i a_b[20], const __m512i b_b[20])
{
    MONT_W16_KARA_X2_BODY(10,
        mont_w16_mul_NxN_x2(z0_a, z0_b, 2*NH, a_a,    b_a,    a_b,    b_b,    NH, NH);
        mont_w16_mul_NxN_x2(z2_a, z2_b, 2*NH, a_a+NH, b_a+NH, a_b+NH, b_b+NH, NH, NH);)
}

static __attribute__((always_inline)) inline
void mont_w16_mul_n40_x2(__m512i c_a[80], __m512i c_b[80],
                           const __m512i a_a[40], const __m512i b_a[40],
                           const __m512i a_b[40], const __m512i b_b[40])
{
    MONT_W16_KARA_X2_BODY(20,
        mont_w16_mul_n20_x2(z0_a, z0_b, a_a,    b_a,    a_b,    b_b);
        mont_w16_mul_n20_x2(z2_a, z2_b, a_a+NH, b_a+NH, a_b+NH, b_b+NH);)
}

static __attribute__((always_inline)) inline
void mont_w16_mul_n80_x2(__m512i c_a[2*MONT_N80], __m512i c_b[2*MONT_N80],
                           const __m512i a_a[MONT_N80], const __m512i b_a[MONT_N80],
                           const __m512i a_b[MONT_N80], const __m512i b_b[MONT_N80])
{
    MONT_W16_KARA_X2_BODY(MONT_N80/2,
        mont_w16_mul_n40_x2(z0_a, z0_b, a_a,    b_a,    a_b,    b_b);
        mont_w16_mul_n40_x2(z2_a, z2_b, a_a+NH, b_a+NH, a_b+NH, b_b+NH);)
}

#endif /* MONT_H */
