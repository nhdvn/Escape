#pragma once
#include "../plhe/params.h"
#include <stdint.h>
#include <stddef.h>

/*
 * db.h -- Virtual database
 *
 * The database is never fully materialized on disk.  Row (k, r) is generated
 * deterministically from SEED_DB and the indices (k, r) via xorshift64.
 * This lets you benchmark against an arbitrarily large logical DB (N*M*M
 * entries) using only a seed -- no large file required.
 *
 * The physical "file" is just the 8-byte seed in PLHE_D_FILE alongside
 * the precomputed dhint D.
 *
 * To support true mmap of a real DB later: replace virtual_db_row() with a call
 * that mmap's a flat binary and returns a pointer to the right offset.
 */

/* Generate physical row of partition k into out[PLHE_MC1 * B_CHUNKS].
 * Layout: interleaved chunks -- out[j*N_CHUNKS + c] is logical col j, chunk c.
 * Entries are in [0, PLHE_P).  k in [0, PLHE_N1), row in [0, PLHE_MR1).  */
static inline void virtual_db_row(int k, int row, uint8_t *out)
{
    uint8_t base = (uint8_t)((uint32_t)k * 65537u + (uint32_t)row * 257u + 1u);
    for (int j = 0; j < PLHE_MC1 * B_CHUNKS; j++)
        out[j] = (uint8_t)(base + (uint8_t)j);
}

/* Byte-ramp fill. Per-byte varying so compiler can't constant-fold heap loads. 
 * Seed derived from the pointer so different buffers get different content. */
static inline void random_buffer(void *buf, size_t n)
{
    uint8_t *p = (uint8_t *)buf;
    uint8_t base = (uint8_t)((uintptr_t)p * 17u + 1u);
    for (size_t j = 0; j < n; j++)
        p[j] = (uint8_t)(base + (uint8_t)j);
}
