/*
 * test_cpu.c -- sanity-check measure.h's cycle / stall / memory decomposition.
 *
 * Runs three single-thread microbenchmarks with the same MEASURE_START /
 * MEASURE_FINIS macros the bench uses, and reports the derived
 *   ans_cycle = (cycles - stall)        // back-end executing
 *   cpu_stall = (stall - memory)        // non-memory stall (ports, deps)
 *   mem_stall = memory                  // L1/L2/L3/DRAM wait
 *
 * Workload                  expectation (qualitative)
 * ----------------------    -------------------------------------------
 * (1) compute_only          mem_stall ~ 0%, cpu_stall small.
 *                           Used to estimate effective freq cycle/wall.
 * (2) memory_only           mem_stall dominant (>50% of cycles).
 *                           Streams a >> L3 buffer at 64 B stride.
 * (3) sleep_only            cycle ~ 0 even though wall ~ N ms.
 *                           perf counts only userspace cycles ON the thread,
 *                           so a descheduled thread should report ~no cycles.
 *
 * Build (from the Escape root):
 *   gcc -O3 -march=native -std=c11 -D_POSIX_C_SOURCE=200112L \
 *       -Iutils -o test_cpu utils/test_cpu.c
 *
 * Pin to a quiet core for the cleanest signal:
 *   taskset -c 1 ./test_cpu
 *
 * Exit code: 0 if all three verdicts hold, 1 otherwise.
 */

#include "measure.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifndef MEM_BYTES
#define MEM_BYTES      (1ULL << 30)    /* 1 GiB, far past any L3            */
#endif
#ifndef COMPUTE_ITERS
#define COMPUTE_ITERS  (1ULL << 30)    /* ~1 billion register IMUL steps    */
#endif
#ifndef SLEEP_MS
#define SLEEP_MS       100
#endif

/* Volatile sinks so -O3 cannot elide the loops. */
static volatile uint64_t sink_u64;

/* ------------------------------------------------------------------ workloads
 */

static void compute_only(void)
{
    /* Register-only LCG step: 1 IMUL + 1 ADD per iter, no memory access.   */
    uint64_t x = 1;
    for (uint64_t i = 0; i < COMPUTE_ITERS; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    sink_u64 = x;
}

static void memory_only(const uint8_t *buf, size_t bytes)
{
    /* Stride by one cache line: ~1 miss per 64 B -> DRAM-bandwidth bound. */
    uint64_t s = 0;
    for (size_t i = 0; i < bytes; i += 64) {
        s ^= buf[i];
    }
    sink_u64 = s;
}

static void sleep_only(void)
{
    struct timespec ts = { 0, (long)SLEEP_MS * 1000000L };
    nanosleep(&ts, NULL);
}

/* ------------------------------------------------------------------ harness
 *
 * Mirror what the bench requires: pc / pc_ok / delay_total /
 * delay_cycle / delay_stall / delay_memory all in scope when invoking
 * MEASURE_START() / MEASURE_FINIS().
 */

typedef struct {
    long     delay_total;
    uint64_t delay_cycle, delay_stall, delay_memory;
} sample_t;

static measure_t pc;
static int       pc_ok;

#define SAMPLE(s, body) do {                                              \
    long     delay_total  = 0;                                            \
    uint64_t delay_cycle  = 0, delay_stall = 0, delay_memory = 0;         \
    MEASURE_START();                                                      \
    body;                                                                 \
    MEASURE_FINIS();                                                      \
    (s).delay_total  = delay_total;                                       \
    (s).delay_cycle  = delay_cycle;                                       \
    (s).delay_stall  = delay_stall;                                       \
    (s).delay_memory = delay_memory;                                      \
} while (0)

static void print_sample(const char *label, sample_t s, double freq_hz)
{
    double wall_ms    = (double)s.delay_total / 1e6;
    double cycle_ms   = freq_hz > 0 ? (double)s.delay_cycle  / freq_hz * 1e3 : 0;
    double ans_cyc_ms = freq_hz > 0 ? (double)(s.delay_cycle - s.delay_stall)  / freq_hz * 1e3 : 0;
    double cpu_stl_ms = freq_hz > 0 ? (double)(s.delay_stall - s.delay_memory) / freq_hz * 1e3 : 0;
    double mem_stl_ms = freq_hz > 0 ? (double)(s.delay_memory) / freq_hz * 1e3 : 0;
    double stl_frac   = s.delay_cycle ? 100.0 * (double)s.delay_stall  / (double)s.delay_cycle : 0;
    double mem_frac   = s.delay_cycle ? 100.0 * (double)s.delay_memory / (double)s.delay_cycle : 0;

    printf("=== %s ===\n", label);
    printf("  wall      %8.1f ms\n", wall_ms);
    printf("  cycles    %8.1f ms  (%llu cycles)\n",
           cycle_ms, (unsigned long long)s.delay_cycle);
    printf("  stall     %8.1f %% of cycles\n", stl_frac);
    printf("  memory    %8.1f %% of cycles\n", mem_frac);
    printf("  -> ans_cycle %.1f ms | cpu_stall %.1f ms | mem_stall %.1f ms\n\n",
           ans_cyc_ms, cpu_stl_ms, mem_stl_ms);
}

/* ------------------------------------------------------------------ main
 */

int main(void)
{
    pc = (measure_t)MEASURE_INIT;
    pc_ok = (measure_init(&pc) == 0);
    printf("measure_init: %s  (fds: cycle=%d stall=%d memory=%d)\n\n",
           pc_ok ? "OK" : "FAIL",
           pc.m_cycle, pc.m_stall, pc.m_memory);
    if (!pc_ok) {
        fprintf(stderr,
                "perf_event_open failed -- check that the user can read\n"
                "perf counters (e.g. /proc/sys/kernel/perf_event_paranoid <= 1).\n");
        return 1;
    }

    /* Warm up: cache fill, freq ramp, BTB. */
    compute_only();

    /* (1) Compute-only -- also gives us an effective-frequency estimate. */
    sample_t s1;
    SAMPLE(s1, compute_only());
    double freq_hz = (s1.delay_total > 0)
        ? (double)s1.delay_cycle / ((double)s1.delay_total / 1e9)
        : 0;
    printf("estimated effective frequency: %.3f GHz\n\n", freq_hz / 1e9);
    print_sample("(1) compute_only", s1, freq_hz);

    /* (2) Memory-only -- pre-touch pages to push first-touch faults out
       of the measured region. */
    uint8_t *buf = aligned_alloc(64, MEM_BYTES);
    if (!buf) { perror("aligned_alloc"); return 1; }
    memset(buf, 0xA5, MEM_BYTES);

    sample_t s2;
    SAMPLE(s2, memory_only(buf, MEM_BYTES));
    print_sample("(2) memory_only (1 GiB streaming)", s2, freq_hz);

    free(buf);

    /* (3) Sleep-only -- thread is descheduled; userspace cycles should ~ 0. */
    sample_t s3;
    SAMPLE(s3, sleep_only());
    print_sample("(3) sleep_only (100 ms nanosleep)", s3, freq_hz);

    /* ----- verdict ----- */
    double f1_mem  = s1.delay_cycle ? (double)s1.delay_memory / (double)s1.delay_cycle : 0;
    double f2_mem  = s2.delay_cycle ? (double)s2.delay_memory / (double)s2.delay_cycle : 0;
    double f3_cyc  = s1.delay_cycle ? (double)s3.delay_cycle  / (double)s1.delay_cycle : 0;

    int ok_compute = (f1_mem < 0.05);
    int ok_memory  = (f2_mem > 0.50);
    int ok_sleep   = (f3_cyc < 0.05);

    printf("--- verdict ---\n");
    printf("  (1) compute  memory frac < 5%%       : %-4s (%.2f%%)\n",
           ok_compute ? "OK" : "FAIL", 100.0 * f1_mem);
    printf("  (2) memory   memory frac > 50%%      : %-4s (%.2f%%)\n",
           ok_memory  ? "OK" : "FAIL", 100.0 * f2_mem);
    printf("  (3) sleep    cycle vs compute < 5%% : %-4s (%llu vs %llu cycles)\n",
           ok_sleep   ? "OK" : "FAIL",
           (unsigned long long)s3.delay_cycle,
           (unsigned long long)s1.delay_cycle);

    measure_close(&pc);
    return (ok_compute && ok_memory && ok_sleep) ? 0 : 1;
}
