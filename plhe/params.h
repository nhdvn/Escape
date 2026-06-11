#pragma once
#include <stdint.h>
#include <stddef.h>

/*
 * Escape v1 -- PIR with server-side dhint.
 *
 * Scheme overview:
 *   Public matrix A: each DB row (k, r) has its own MC1xLambda matrix A[k,r],
 *   generated on demand from SEED_A1 via AES-128 CTR with the counter
 *   binding (db_row_flat = k*MR1+r, col_idx).
 *
 *   Database DB: N1 partitions, each MR1xMC1, entries in Z_p.
 *   In production the server precomputes and stores dhint = DB*A; this
 *   bench regenerates DB + dhint rows on the fly each query (deterministic
 *   from the same A seed) to avoid the storage cost on a large database.
 *
 *   Per query, single secret key sk in {0,1}^Lambda, shared across all partitions.
 *   For each partition k, client encrypts the desired column c_k:
 *     enc_bits[k][j] = A[k,sub_rows[k]][j,:]*sk + e_j + Delta*[j == c_k]
 *     sub_rows[k] sent in the clear.
 *
 *   Server (per query):
 *     ans[c] = Sum_k Sum_j DB[k,sub_rows[k],j,c] * enc_bits[k][j]
 *     agg[c] = Sum_k dhint[k,sub_rows[k],:,c]   (Lambda-vector per chunk)
 *
 *   Client (recovered via Paillier decompress_response):
 *     plain[c] = round( (ans[c] - agg[c]*sk) / Delta ) mod p
 *              ~= Sum_k DB[k,sub_rows[k],c_k,c]
 *
 *   Sharing one sk across partitions is what makes Sum_k dhint[k,sub_rows[k],:]*sk
 *   factor as (Sum_k dhint[k,sub_rows[k],:])*sk -- i.e. the server can pre-aggregate
 *   without knowing sk and let the client finish the inner product.
 *
 * All parameters in this file are the only place you need to change to
 * reconfigure the scheme.
 */

/* -- Ciphertext modulus ----------------------------------------------------
 * Q_BITS=32: uint32_t; overflow acts as mod 2^32 -- no masking needed.
 * Q_BITS=64: uint64_t; overflow acts as mod 2^64.                           */
#define PLHE_Q_BITS    32
typedef uint32_t       plhe_q_t;

/* -- Plaintext modulus: p = 2^P_BITS -------------------------------------*/
#define PLHE_P_BITS    8
#define PLHE_P         (1u << PLHE_P_BITS)
#define PLHE_P_MASK    ((uint32_t)(PLHE_P - 1u))

/* -- Scale factor Delta = q/p = 2^(Q_BITS - P_BITS) --------------------------*/
#define PLHE_DELTA     ((plhe_q_t)1 << (PLHE_Q_BITS - PLHE_P_BITS))

/* -- LWE secret key dimension --------------------------------------------*/
#ifndef PLHE_LAMBDA
#define PLHE_LAMBDA    1024
#endif

/* -- Database layout -----------------------------------------------------
 * N1 partitions, each an MR1xMC1 sub-matrix.
 * Total entries = N1*MR1*MC1.                                             */
#ifndef PLHE_N1
#define PLHE_N1        256    /* number of partitions                     */
#endif
#ifndef PLHE_MR1
#define PLHE_MR1       256    /* rows per partition (row_index range)     */
#endif
#ifndef PLHE_MC1
#define PLHE_MC1       256    /* columns per partition (ciphertext size)  */
#endif

/* -- LWE error standard deviation x 10 (integer; avoids float #define) --*/
#ifndef PLHE_SIGMA_X10
#define PLHE_SIGMA_X10 32
#endif

/* -- Worker threads ------------------------------------------------------
 * Server: SERVER_THREADS workers, each owning N1/SERVER_THREADS partitions.
 * Each worker processes ks in groups of AGG_BATCH: gen AGG_BATCH (db,r_hint),
 * FMA each, then a single fused agg-add that reads the agg cache line
 * once and adds AGG_BATCH r_hints. Cuts agg-RMW DRAM traffic from
 *   3 ops/line/k  ->  (AGG_BATCH+2) ops/line/AGG_BATCH   (~AGG_BATCH/3 reduction).
 * Cost: AGG_BATCH * B_CHUNKS * LAMBDA * 4 bytes of r_hint scratch per worker. */
#ifndef SERVER_THREADS
#define SERVER_THREADS   36
#endif
#ifndef AGG_BATCH
#define AGG_BATCH        8
#endif
#ifndef CLIENT_THREADS
#define CLIENT_THREADS   3
#endif

/* -- Block size and chunk count ------------------------------------------
 * BLOCK_KIB: integer block size in KiB (entry size in DB).
 * BLOCK_SIZE_BITS: rounded up to make B_CHUNKS even (SIMD-friendly).
 * B_CHUNKS: how many P_BITS-bit PLHE queries to retrieve one entry.        */
#ifndef BLOCK_KIB
#define BLOCK_KIB        4
#endif
#ifndef BLOCK_SIZE_BITS
#define BLOCK_SIZE_BITS \
    (((((BLOCK_KIB * 8192) + PLHE_P_BITS - 1) / PLHE_P_BITS + 1) / 2 * 2) * PLHE_P_BITS)
#endif
#define B_CHUNKS          ((BLOCK_SIZE_BITS + PLHE_P_BITS - 1) / PLHE_P_BITS)

/* -- Deterministic seeds (both client and server derive from these) ------*/
#define SEED_DB   UINT64_C(0x0123456789abcdef)   /* virtual database  */
#define SEED_A1   UINT64_C(0xdeadbeefcafebabe)   /* matrix A          */

/* -- Network bandwidth assumed for [client send]/[client recv] wire-time
 *    estimates printed by the bench. Doesn't affect any computation --
 *    purely a display unit for "x bytes -> y ms @ Mbps".               */
#ifndef NET_MBPS
#define NET_MBPS  70
#endif

/* -- Bench-only: skip data-generation cost -------------------------------
 * When set to 1, the server worker skips virtual_db_row() and plhe_gen_A_row()
 * and uses zero-filled buffers of the same size. The dhint+ans mul-add
 * pipeline still runs with the same memory access pattern, so this isolates
 * the accumulate cost from the AES-CTR generation cost.
 *
 * Correctness will FAIL in this mode (results won't match expected). For
 * correctness verification, recompile with -DBENCH_SKIP_GEN=0.        */
#ifndef BENCH_SKIP_GEN
#define BENCH_SKIP_GEN 0
#endif
