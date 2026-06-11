#pragma once
#include <pthread.h>
#include <stdatomic.h>
#include <immintrin.h>

typedef struct {
    _Atomic int count;
    _Atomic int gen;
    int         trip;
} barrier_t;

static inline void barrier_init(barrier_t *b, int trip)
{
    atomic_init(&b->count, 0);
    atomic_init(&b->gen,   0);
    b->trip = trip;
}

static inline void barrier_destroy(barrier_t *b) { (void)b; }

static inline void barrier_wait(barrier_t *b)
{
    int gen = atomic_load_explicit(&b->gen, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) + 1 == b->trip) {
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&b->gen, 1, memory_order_release);
    } else {
        while (atomic_load_explicit(&b->gen, memory_order_acquire) == gen)
            _mm_pause();
    }
}
