#pragma once
#include "../plhe/params.h"
#include "../channel/channel.h"

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

typedef struct server_s server_t;

server_t *server_create(void);
void      server_destroy(server_t *server);

void server_answer(server_t *server, channel_t *channel);

/* ---- Server-side SIMD helpers used by worker_fn / server_answer -------- */

/* dst[lc] += zext(src_u8[lc]) * scalar, unaligned + scalar tail. */
#define U8_FMA_LOOP(dst, src, scalar, n)                                   \
    ({                                                                     \
        __m512i _vct = _mm512_set1_epi32((int)(scalar));                   \
        int _lc = 0;                                                       \
        for (; _lc + 16 <= (n); _lc += 16) {                               \
            __m512i _vs  = _mm512_cvtepu8_epi32(                           \
                _mm_loadu_si128((const __m128i *)((src) + _lc)));          \
            __m512i _vd  = _mm512_loadu_si512((dst) + _lc);                \
            _mm512_storeu_si512((dst) + _lc,                               \
                _mm512_add_epi32(_vd, _mm512_mullo_epi32(_vs, _vct)));     \
        }                                                                  \
        for (; _lc < (n); _lc++)                                           \
            (dst)[_lc] += (uint32_t)(src)[_lc] * (uint32_t)(scalar);       \
    })

/* dst[i] += src[i]. dst, src 64-byte aligned, n a multiple of 16. */
#define U32_ADD_LOOP_ALIGNED(dst, src, n)                                  \
    ({                                                                     \
        for (int _i = 0; _i < (n); _i += 16) {                             \
            __m512i _va = _mm512_load_si512((dst) + _i);                   \
            __m512i _vd = _mm512_load_si512((src) + _i);                   \
            _mm512_store_si512((dst) + _i, _mm512_add_epi32(_va, _vd));    \
        }                                                                  \
    })

/* dst[i] += sum_{b=0..nsrc-1} srcs[b][i]. All 64-byte aligned, n mult of 16. */
#define U32_ADDN_LOOP_ALIGNED(dst, srcs, nsrc, n)                          \
    ({                                                                     \
        for (int _i = 0; _i < (n); _i += 16) {                             \
            __m512i _va = _mm512_load_si512((dst) + _i);                   \
            for (int _b = 0; _b < (nsrc); _b++) {                          \
                __m512i _vs = _mm512_load_si512((srcs)[_b] + _i);          \
                _va = _mm512_add_epi32(_va, _vs);                          \
            }                                                              \
            _mm512_store_si512((dst) + _i, _va);                           \
        }                                                                  \
    })

/* 64-byte aligned, zero-filled alloc (cache-line / zmm). */
#define ALLOC_ALIGN64(ptr, n_bytes)                                        \
    ({                                                                     \
        size_t _nb = (size_t)(n_bytes);                                    \
        (void)posix_memalign((void **)&(ptr), 64, _nb);                    \
        memset((ptr), 0, _nb);                                             \
    })
