#pragma once

/*
 * measure.h -- custom wrapper over perf_event_open for per-thread cycle
 * counters used to break a wall-clock timer into compute vs stall cycle.
 *
 * Decomposition:
 *   compute_time   = cycles - stall            (back-end execution)
 *   compute_stall  = stall - memory            (latency/port stall)
 *   memory_stall   = memory                    (L1/L2/L3/DRAM wait)
 *   wall ~ cycles / effective_freq
 *
 * Open events once per thread, then bracket each timed region with
 * measure_read() to accumulate delta counts.  The events are
 * always-on between read() calls; we just diff the running total.
 */

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Declare syscall manually -- _POSIX_C_SOURCE strict mode hides it.    */
extern long syscall(long number, ...);

typedef struct {
    int m_cycle;          /* HW_CPU_CYCLES                  */
    int m_stall;          /* CYCLE_ACTIVITY.STALLS_TOTAL    */
    int m_memory;         /* CYCLE_ACTIVITY.STALLS_MEM_ANY  */
} measure_t;

#define MEASURE_INIT { -1, -1, -1 }

/* Set (type, config) on `a` and perf_event_open on the current thread;
 * returns the fd or -1. Used to walk a fallback chain across uarchs. */
static inline int _try_event_(struct perf_event_attr *a,
                                      uint32_t type, uint64_t config)
{
    a->type   = type;
    a->config = config;
    return (int)syscall(__NR_perf_event_open, a, 0, -1, -1, 0);
}

/* CPU vendor (CPUID leaf 0). PERF_TYPE_RAW config bits are interpreted by
 * the host PMU driver in its native format. Same hex can refer to distinct
 * events across vendors -- gate raw fallbacks by vendor to avoid garbage. */

enum { _MEAS_VENDOR_UNKNOWN = 0, _MEAS_VENDOR_INTEL = 1, _MEAS_VENDOR_AMD = 2 };

static inline int _measure_vendor_(void)
{
#if defined(__x86_64__) || defined(__i386__)
    unsigned eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(0));
    /* "GenuineIntel" */
    if (ebx == 0x756e6547u && ecx == 0x6c65746eu && edx == 0x49656e69u)
        return _MEAS_VENDOR_INTEL;
    /* "AuthenticAMD" */
    if (ebx == 0x68747541u && ecx == 0x444d4163u && edx == 0x69746e65u)
        return _MEAS_VENDOR_AMD;
#endif
    return _MEAS_VENDOR_UNKNOWN;
}

/* Open all three counters on the current thread.  Returns 0 on success,
 * -1 on hard failure (m_cycle or m_stall couldn't be opened).  m_memory
 * is best-effort -- left at -1 if no encoding works (then memory_cycles
 * reads as 0 and "exec stall" can't be separated from total stall).
 *
 * m_cycle : PERF_TYPE_HARDWARE / HW_CPU_CYCLES                  (universal)
 *
 * m_stall : back-end stall, in priority order:
 *    (a) HW_STALLED_CYCLES_BACKEND  (generic kernel maps to per-uarch event)
 *    (b) Intel fallbacks:
 *        RAW 0x040004a3 = CYCLE_ACTIVITY.STALLS_TOTAL  (SNB+ Intel)
 *        RAW 0x000001a2 = RESOURCE_STALLS.ANY          (SNB..SPR Intel)
 *    (c) AMD fallback:
 *        RAW 0x0000FFAE = De_Dis_Dispatch_Stalls.Any   (Zen, all sub-causes)
 *
 * m_memory : memory-subsystem stall, with vendor-specific encodings.
 *            If nothing works: no memory-vs-compute stall split.
 *    Intel : STALLS_MEM_ANY (Ice Lake+) -> STALLS_LDM_PENDING (Haswell+) 
 *            -> RESOURCE_STALLS.SB (store-buffer weak proxy).
 *    AMD   : MAB_Alloc.All -- cycles with at least one outstanding L1D miss
 *            (event 0x47, umask 0x3F over load/store sub-streams). AMD has
 *            no direct counterpart of Intel STALLS_MEM_ANY (which gates the
 *            "actually stalled" AND "memory request in flight" condition).
 *            This proxy counts cycles where a miss is in flight regardless
 *            of whether dispatch stalled on it -- a strict upper bound for
 *            memory_stall. Therefore (m_stall - m_memory) is a lower bound
 *            on non-memory stall on AMD. The qualitative memory-vs-compute
 *            verdict is reliable. The absolute split is fuzzier than Intel.    */

 static inline int measure_init(measure_t *p)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    a.disabled       = 0;

    int vendor = _measure_vendor_();

    /* TRY(field, type, config): Attempt IF `field` is still -1. */
    #define TRY(F, T, C) ({ if ((F) < 0) (F) = _try_event_(&a, (T), (C)); })
    #define TRY_RAW(F, C) TRY((F), PERF_TYPE_RAW, (C))

    TRY(p->m_cycle, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    if (p->m_cycle < 0) { p->m_cycle = -1; return -1; }

    TRY(p->m_stall, PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
    if (vendor == _MEAS_VENDOR_INTEL) {
        TRY_RAW(p->m_stall, 0x040004a3ULL);  /* CYCLE_ACTIVITY.STALLS_TOTAL */
        TRY_RAW(p->m_stall, 0x000001a2ULL);  /* RESOURCE_STALLS.ANY         */
    } else if (vendor == _MEAS_VENDOR_AMD) {
        TRY_RAW(p->m_stall, 0x0000FFAEULL);  /* De_Dis_Dispatch_Stalls.Any  */
    }

    if (p->m_stall < 0) {
        close(p->m_cycle);
        p->m_cycle = -1;
        return -1;
    }

    if (vendor == _MEAS_VENDOR_INTEL) {
        TRY_RAW(p->m_memory, 0x140014a3ULL); /* CYCLE_ACTIVITY.STALLS_MEM_ANY       */
        TRY_RAW(p->m_memory, 0x060006a3ULL); /* CYCLE_ACTIVITY.STALLS_LDM_PENDING   */
        TRY_RAW(p->m_memory, 0x000008a2ULL); /* RESOURCE_STALLS.SB                  */
    } else if (vendor == _MEAS_VENDOR_AMD) {
        TRY_RAW(p->m_memory, 0x00003F47ULL); /* MAB_Alloc.All           */
    }

    #undef TRY_RAW
    #undef TRY
    return 0;
}

static inline void measure_close(measure_t *p)
{
    if (p->m_cycle  >= 0) close(p->m_cycle);
    if (p->m_stall  >= 0) close(p->m_stall);
    if (p->m_memory >= 0) close(p->m_memory);
    p->m_cycle = p->m_stall = p->m_memory = -1;
}

static inline void measure_read(const measure_t *p,
                                     uint64_t *cycle, uint64_t *stall, uint64_t *memory)
{
    *cycle = *stall = *memory = 0;
    if (p->m_cycle < 0) return;
    if (read(p->m_cycle,  cycle,  sizeof(*cycle))  != (ssize_t)sizeof(*cycle))  *cycle = 0;
    if (read(p->m_stall,  stall,  sizeof(*stall))  != (ssize_t)sizeof(*stall))  *stall = 0;
    if (p->m_memory >= 0 &&
        read(p->m_memory, memory, sizeof(*memory)) != (ssize_t)sizeof(*memory)) *memory = 0;
}

/* MEASURE_START/MEASURE_FINIS: bracket a hot path.  Expects in caller scope:
 *   measure_t pc;          int pc_ok;
 *   long delay_total;      uint64_t delay_cycle, delay_stall, delay_memory;
 * The wall-clock now_ns() is provided above. */

#define MEASURE_START()                                             \
    uint64_t _c0 = 0, _s0 = 0, _m0 = 0;                             \
    uint64_t _t0 = now_ns();                                        \
    if (pc_ok) measure_read(&pc, &_c0, &_s0, &_m0)

#define MEASURE_FINIS()                                             \
    ({                                                              \
        delay_total += (long)(now_ns() - _t0);                      \
        if (pc_ok) {                                                \
            uint64_t _c1, _s1, _m1;                                 \
            measure_read(&pc, &_c1, &_s1, &_m1);                    \
            delay_cycle  += _c1 - _c0;                              \
            delay_stall  += _s1 - _s0;                              \
            delay_memory += _m1 - _m0;                              \
        }                                                           \
    })
