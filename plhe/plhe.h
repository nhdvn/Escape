#pragma once
#include "params.h"
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

/*
 * plhe.h -- Escape v1 PLHE primitives.
 *
 * Array dimensions (row-major):
 *   sk         : [LAMBDA]              uint8_t  -- binary secret in {0,1}
 *   enc_bits   : [MC1]                 plhe_q_t -- column-selector ciphertext
 *   dh_row     : [B_CHUNKS * LAMBDA]   plhe_q_t -- one row of dhint = DB * A
 *   db_row     : [MC1 * B_CHUNKS]      uint8_t  -- one DB row (B_CHUNKS values per col)
 */

/* -- AES-128 CTR PRNG ----------------------------------------------------
 * Two key schedules, initialized at program start (constructor in plhe.c):
 *   _A1_rk  (static): generates A from SEED_A1
 *   plhe_rng_rk (extern): secret/error sampling, shared across modules    */
extern __m128i plhe_rng_rk[11];

typedef struct {
    __m128i  ctr;
    uint64_t cached;
    int      has_cached;
} plhe_rng_t;

static inline void plhe_rng_init(plhe_rng_t *r, uint64_t seed)
{
    r->ctr        = _mm_set_epi64x(0LL, (long long)seed);
    r->has_cached = 0;
}

static inline __m128i aes128_block(const __m128i *rk, __m128i ctr)
{
    __m128i x = _mm_xor_si128(ctr, rk[0]);
    x = _mm_aesenc_si128(x, rk[1]);
    x = _mm_aesenc_si128(x, rk[2]);
    x = _mm_aesenc_si128(x, rk[3]);
    x = _mm_aesenc_si128(x, rk[4]);
    x = _mm_aesenc_si128(x, rk[5]);
    x = _mm_aesenc_si128(x, rk[6]);
    x = _mm_aesenc_si128(x, rk[7]);
    x = _mm_aesenc_si128(x, rk[8]);
    x = _mm_aesenc_si128(x, rk[9]);
    return _mm_aesenclast_si128(x, rk[10]);
}

static inline uint64_t plhe_rng(plhe_rng_t *r)
{
    if (r->has_cached) { r->has_cached = 0; return r->cached; }
    __m128i x     = aes128_block(plhe_rng_rk, r->ctr);
    r->ctr        = _mm_add_epi64(r->ctr, _mm_set_epi64x(0LL, 1LL));
    r->cached     = (uint64_t)_mm_extract_epi64(x, 0);
    r->has_cached = 1;
    return          (uint64_t)_mm_extract_epi64(x, 1);
}

/* -- AVX-512 VAES 8-chain pipeline macros (used by plhe_gen_A_row /
 *    plhe_encrypt_vbit).  CTR_* build counter blocks; VAES_LOOP iterates
 *    32 blocks per round, applying APPLY8 fan-out to the 8-deep AES chain.
 *    FINAL_OP picks the per-iteration tail (store vs masked-add). */

/* APPLY8 leaves declarations in the enclosing scope (no do-while wrap). */
#define APPLY8(OP)        OP(0); OP(1); OP(2); OP(3); OP(4); OP(5); OP(6); OP(7)
#define CTR_LO_BASE       _mm512_set_epi64(0, 3, 0, 2, 0, 1, 0, 0)
#define CTR_LO_STEP       _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4)
#define CTR_HI_LANES(hi)  _mm512_set_epi64((hi), 0, (hi), 0, (hi), 0, (hi), 0)
#define CTR_HI(row, col)  (((long long)(uint32_t)(row) << 32) | (long long)(uint32_t)(col))

#define CTR_CHAIN_FROM_B(b)                                               \
    __m512i ctr0 = _mm512_or_si512(hi_lanes,                              \
        _mm512_add_epi64(_mm512_set1_epi64((long long)(b)), lo_base));    \
    __m512i ctr1 = _mm512_add_epi64(ctr0, lo_step);                       \
    __m512i ctr2 = _mm512_add_epi64(ctr1, lo_step);                       \
    __m512i ctr3 = _mm512_add_epi64(ctr2, lo_step);                       \
    __m512i ctr4 = _mm512_add_epi64(ctr3, lo_step);                       \
    __m512i ctr5 = _mm512_add_epi64(ctr4, lo_step);                       \
    __m512i ctr6 = _mm512_add_epi64(ctr5, lo_step);                       \
    __m512i ctr7 = _mm512_add_epi64(ctr6, lo_step)

#define VAES_INIT_X(i)  __m512i x##i = _mm512_xor_si512(ctr##i, _A1_zk[0])
#define VAES_AENC_X(i)  x##i = _mm512_aesenc_epi128(x##i, _A1_zk[r])
#define VAES_LAST_X(i)  x##i = _mm512_aesenclast_epi128(x##i, _A1_zk[10])
#define VAES_STORE_X(i) _mm512_storeu_si512((__m512i *)(out) + b/4 + i, x##i)
#define VAES_MKADD_X(i) sum = _mm512_mask_add_epi32(sum, mask[b/4 + i], sum, x##i)

/* FINAL_OP: VAES_STORE_X | VAES_MKADD_X. n_blocks must be a multiple of 32. */
#define VAES_LOOP(n_blocks_, FINAL_OP)                                     \
    ({                                                                     \
        for (int b = 0; b < (n_blocks_); b += 32) {                        \
            CTR_CHAIN_FROM_B(b);                                           \
            APPLY8(VAES_INIT_X);                                           \
            for (int r = 1; r < 10; r++) { APPLY8(VAES_AENC_X); }          \
            APPLY8(VAES_LAST_X);                                           \
            APPLY8(FINAL_OP);                                              \
        }                                                                  \
    })

/* -- Secret key -----------------------------------------------------------
 * sk holds PLHE_LAMBDA binary coefficients in {0, 1}.
 * One sk is shared across all partitions for a query.                  */
void plhe_keygen(uint8_t *sk, plhe_rng_t *rng);

/* -- Matrix A (row-dependent) --------------------------------------------
 * Each DB row (k, r) uses its OWN independent MC1xLAMBDA matrix A.
 * A is never materialized -- rows generated on demand via AES-128 CTR keyed
 * by SEED_A1, with the counter binding (p_row, col_idx).
 *
 *   p_row      = k * PLHE_MR1 + r   (unique ID for the DB row)
 *   col_idx    in [0, PLHE_MC1)      (A-row index = DB column)
 *   out        must hold PLHE_LAMBDA plhe_q_t, 64-byte aligned.        */
void plhe_gen_A_row(plhe_q_t *out, int p_row, int col_idx);

/* -- Encryption ----------------------------------------------------------
 * For each j in [0, PLHE_MC1):
 *   enc_bits[j] = A[db_row_flat][j,:]*sk + e_j + Delta*[j == col]
 * enc_bits must hold PLHE_MC1 elements.                                */
void plhe_encrypt_vbit(plhe_q_t *enc_bits, int p_row, int col,
                       const uint8_t *sk, plhe_rng_t *rng);

/* -- dhint row ------------------------------------------------------------
 * Compute B_CHUNKS dhint vectors from one physical DB row (k, r):
 *   dhint_rows[c*LAMBDA + l] = Sum_j db_row_phys[j*B_CHUNKS+c]
 *                              * A[db_row_flat][j*LAMBDA + l]
 * dhint_rows must hold B_CHUNKS * PLHE_LAMBDA elements, 64-byte aligned.*/
void plhe_dhint_row(plhe_q_t *dh_row, const uint8_t *db_row, int p_row);
