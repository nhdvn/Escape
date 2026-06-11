#include "plhe.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

_Static_assert((PLHE_LAMBDA * 4) % 512 == 0,
               "PLHE_LAMBDA must be aligned for VAES 32-block tile");

#define GAUSS_BOUND      20                    /* 6.25 for sigma 3.2 */
#define GAUSS_TABLE_SIZE (2 * GAUSS_BOUND)

static uint64_t gauss_cdt[GAUSS_TABLE_SIZE];

static void init_gauss_cdt(void)
{
    double sigma    = (double)PLHE_SIGMA_X10 / 10.0;
    double _2sigma  = 2.0 * sigma * sigma;

    double w[GAUSS_BOUND + 1];
    for (int k = 0; k <= GAUSS_BOUND; k++)
        w[k] = exp(-(double)(k * k) / _2sigma);
    double Z = w[0];
    for (int k = 1; k <= GAUSS_BOUND; k++) Z += 2.0 * w[k];

    double gsum = 0.0;
    for (int i = 0; i < GAUSS_TABLE_SIZE; i++) {
        int k = i - GAUSS_BOUND;
        gsum += w[k < 0 ? -k : k] / Z;
        gauss_cdt[i] = (uint64_t)(gsum * (double)UINT64_MAX);
    }
}

static inline int64_t sample_gauss(plhe_rng_t *rng)
{
    uint64_t u = plhe_rng(rng);
    int x = -GAUSS_BOUND;
    for (int i = 0; i < GAUSS_TABLE_SIZE; i++)
        x += (int)(u >= gauss_cdt[i]);
    return (int64_t)x;
}

static __m128i _A1_rk[11];
static __m512i _A1_zk[11];     /* zmm-broadcast of _A1_rk */
__m128i        plhe_rng_rk[11];

static inline __m128i key_expand(__m128i k, __m128i kg)
{
    kg = _mm_shuffle_epi32(kg, 0xff);
    k  = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k  = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k  = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, kg);
}

static void aes128_key_schedule(__m128i rk[11], __m128i k)
{
    static const uint8_t rcon[10] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36 };
    rk[0] = k;
    for (int i = 0; i < 10; i++)
        rk[i+1] = key_expand(rk[i], _mm_aeskeygenassist_si128(rk[i], rcon[i]));
}

static void __attribute__((constructor)) aes_key_setup(void)
{
    aes128_key_schedule(_A1_rk,
        _mm_set_epi64x((long long)(SEED_A1 ^ UINT64_C(0xdeadc0de00000000)),
                       (long long)SEED_A1));
    aes128_key_schedule(plhe_rng_rk,
        _mm_set_epi64x((long long)(SEED_A1 ^ UINT64_C(0x900dcafe00000000)),
                       (long long)(SEED_A1 ^ UINT64_C(0xffffffffffffffff))));
    for (int i = 0; i < 11; i++)
        _A1_zk[i] = _mm512_broadcast_i32x4(_A1_rk[i]);
    init_gauss_cdt();
}

/* AES-CTR stream keyed on (p_row, col_idx); counter lo64 = block idx. */
void plhe_gen_A_row(plhe_q_t *out, int p_row, int col_idx)
{
    long long hi = CTR_HI(p_row, col_idx);
    int n_blocks = (int)(PLHE_LAMBDA * sizeof(plhe_q_t) / 16);

    const __m512i hi_lanes = CTR_HI_LANES(hi);
    const __m512i lo_base  = CTR_LO_BASE;
    const __m512i lo_step  = CTR_LO_STEP;

    VAES_LOOP(n_blocks, VAES_STORE_X);
}

void plhe_keygen(uint8_t *sk, plhe_rng_t *rng)
{
    for (int i = 0; i < PLHE_LAMBDA; ) {
        uint64_t r = plhe_rng(rng);
        for (int b = 0; b < 64 && i < PLHE_LAMBDA; b++, r >>= 1)
            sk[i++] = (uint8_t)(r & 1);
    }
}

/* AES gen fused with masked-add dot; A_row stays in zmm regs. */
void plhe_encrypt_vbit(plhe_q_t *enc_bits, int p_row, int col,
                       const uint8_t *sk, plhe_rng_t *rng)
{
    __mmask16 mask[PLHE_LAMBDA / 16];
    for (int i = 0; i < PLHE_LAMBDA; i += 16) {
        __m128i sk16 = _mm_loadu_si128((const __m128i *)(sk + i));
        mask[i / 16] = _mm_cmpneq_epi8_mask(sk16, _mm_setzero_si128());
    }

    const int n_blocks = (int)(PLHE_LAMBDA * sizeof(plhe_q_t) / 16);
    const __m512i lo_base = CTR_LO_BASE;
    const __m512i lo_step = CTR_LO_STEP;

    for (int j = 0; j < PLHE_MC1; j++) {
        long long hi = CTR_HI(p_row, j);
        const __m512i hi_lanes = CTR_HI_LANES(hi);

        __m512i sum = _mm512_setzero_si512();
        VAES_LOOP(n_blocks, VAES_MKADD_X);
        plhe_q_t dot = (plhe_q_t)_mm512_reduce_add_epi32(sum);
        enc_bits[j] = dot + (plhe_q_t)sample_gauss(rng);
    }
    enc_bits[col] += PLHE_DELTA;
}

/* dh[i] = Σ_col db[col] · A[p_row][col][i]    (C = MC1, B_CHUNKS = 1)
 *
 * A[p_row] is conceptually C × λ but never stored: each col iteration
 * regenerates one λ-row via AES-CTR(p_row, col), axpy its scaled copy
 * into dh, then discards it.
 *
 *    Ap_1  =        Ap_1[1]         Ap_1[2]     ⋯       Ap_1[λ]
 *    Ap_2  =        Ap_2[1]         Ap_2[2]     ⋯       Ap_2[λ]
 *      ⋮
 *    Ap_C  =        Ap_C[1]         Ap_C[2]     ⋯       Ap_C[λ]
 *
 *     db   =  db[1]           db[2]           ⋯   db[C]
 *
 *     dh   =  db[1]·Ap_1[1]   db[1]·Ap_1[2]   ⋯   db[1]·Ap_1[λ]
 *                  +               +                   +
 *             db[2]·Ap_2[1]   db[2]·Ap_2[2]   ⋯   db[2]·Ap_2[λ]
 *
 *                  +               +                   +
 *             db[C]·Ap_C[1]   db[C]·Ap_C[2]   ⋯   db[C]·Ap_C[λ]
 *
 * Only one Ap_col (λ words) is resident at a time, never the full
 * C × λ matrix. dh is built column-wise: each col adds one row of
 * scaled products into the running λ-sum. */

void plhe_dhint_row(plhe_q_t *dh_row, const uint8_t *db_row, int p_row)
{
    memset(dh_row, 0, (size_t)B_CHUNKS * PLHE_LAMBDA * sizeof(plhe_q_t));
    plhe_q_t A_row[PLHE_LAMBDA] __attribute__((aligned(64)));
    for (int col = 0; col < PLHE_MC1; col++) {
        plhe_gen_A_row(A_row, p_row, col);
        for (int chunk = 0; chunk < B_CHUNKS; chunk++) {
            plhe_q_t  db_item = (plhe_q_t)db_row[col * B_CHUNKS + chunk];
            plhe_q_t *r_chunk = dh_row + (size_t)chunk * PLHE_LAMBDA;
            for (int i = 0; i < PLHE_LAMBDA; i++)
                r_chunk[i] += db_item * A_row[i];
        }
    }
}
