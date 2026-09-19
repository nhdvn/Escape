# ESCAPE: Efficient Single-Server Online-Offline PIR without Periodic Preprocessing (IEEE S&P 2027)

This is the full implementation of our work ESCAPE. The algorithm details can be found in the paper (https://eprint.iacr.org/2026/2033.pdf)

WARNING: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.



# Citing

If the code is found useful, we would be appreciated if our paper can be cited with the following bibtex format:

```
@INPROCEEDINGS{nguyen2027Escape,
  author={Nguyen, HD and Guajardo, Jorge and Hoang, Thang},
  booktitle = {2027 IEEE Symposium on Security and Privacy},
  title = {{Efficient Single-Server Online-Offline PIR without Periodic Preprocessing}},
  year = {2027}
}
```

<br/>



## Prerequisites

- Linux, x86-64 CPU with **AVX-512 IFMA52** (plus AVX-512F/DQ and AES-NI),
  e.g. Intel Ice Lake-SP or newer
- GCC 9+ with `-march=native`, GNU Make
- `libgmp-dev` (or distro equivalent), `libgomp` (ships with GCC)
- Python 3.8+ for the scripts in `scripts/` and `comparison/`

The results in the paper use the AVX-512 IFMA52 code path
(`compress/_m512.c`). On a CPU without IFMA52 the Makefile still builds a GMP
fallback (`compress/_gmpz.c`), but its timings do not correspond to the paper.

Quick check:
```sh
gcc --version                                   # >= 9
ldconfig -p | grep gmp                          # libgmp.so.10 present
grep -o -w avx512ifma /proc/cpuinfo | head -1   # must print avx512ifma
```

Full hardware and software requirements, including the versions we tested
with, are listed in [`metadata.toml`](metadata.toml).

## Build

```sh
make           # builds ./bench with default params
make clean     # remove build/ and binaries
```

Defaults live in [`plhe/params.h`](plhe/params.h):
`N1 = MR1 = MC1 = 256`, `LAMBDA = 1024`, `BLOCK_KIB = 4`, sigma `3.2`,
which describes a logical 64 GiB database (16 M entries × 4 KiB each).

## Reproducing The Paper Results

The comparison in the paper is produced in three steps, run in this order
from the repository root. Step 1 measures Escape, step 2 measures the
baselines, and step 3 extrapolates the baselines to Escape's settings and
writes the plot points.

### Step 1: Measure Escape (`scripts/bench.sh`)

```sh
scripts/bench.sh                          # writes result/
OUT_BASE=result_rerun scripts/bench.sh    # or write to another folder
```

The script rebuilds and runs `./bench 5` for 28 settings (entry size ×
database size), in skip-gen mode (see below):

| entry size | database sizes N |
|------------|------------------|
| 4 KiB      | 2^28 – 2^34      |
| 8 KiB      | 2^26 – 2^32      |
| 16 KiB     | 2^26 – 2^32      |
| 64 KiB     | 2^24 – 2^30      |

Each run writes `result/<entry>/db_log2_<log2N>.log` (for example
`result/64kb/db_log2_30.log`) and appends a summary with the means over the 5
measured repetitions: bandwidth, server delay, the compute / memory-stall
split of the answer and compress stages, and the end-to-end delay. Each log
ends with `FAIL: <N> error(s)`; this is expected in skip-gen mode.

The server uses 36 threads. The largest settings (64 KiB entries) need about
82 GiB of RAM, and the whole sweep takes tens of minutes because every
setting is rebuilt.

### Step 2: Measure The Baselines (`comparison/measure.py`)

```sh
cd comparison
python3 measure.py
```

The script downloads the baselines from GitHub into `comparison/` and builds
them:

- **SimplePIR** (`ahenzinger/simplepir`)
- **InsPIRe** (`google/private-membership/research/InsPIRe`)
- **VIA** (`owniai/VIA`, which ships its own Intel HEXL library)

It also uses **OSimplePIR**, our OpenMP build of SimplePIR, which is part of
this repository. It then measures each scheme on 2^14, 2^16 and 2^18 entries
of 4 KiB, 5 trials each: SimplePIR, InsPIRe and VIA single-threaded, and
OSimplePIR with 12, 24 and 36 threads. It needs internet access and these
extra tools: git, Go 1.21+, Rust via rustup (InsPIRe pins its own nightly
toolchain, which rustup installs automatically), and g++ with C++17 and
AVX-512.

Outputs:

- `comparison/Measure/`: one log per run, emptied at the start of every run.
- `comparison/summary.log`: first-pass throughput (GiB/s) for every scheme,
  size and thread count. The previous `summary.log` is moved to
  `summary.prev`. The file ends with an **Optimistic** section: the mean of
  each scheme's 3 best first-pass throughputs, the ratio OSimplePIR /
  SimplePIR, and the multi-threaded estimates OInsPIRe and OVIA (the
  single-threaded InsPIRe and VIA throughput times that ratio).

The run takes about 20 minutes, mostly for generating the 2^18 databases.

### Step 3: Extrapolate And Write Plot Points (`comparison/extrapolate.py`)

```sh
cd comparison
python3 extrapolate.py generate     # per-setting breakdown of every scheme
python3 extrapolate.py bandwidth    # plot points: upload + download (MiB)
python3 extrapolate.py compute      # plot points: query generation + server (s)
```

The script takes the six throughputs from the Optimistic section of
`comparison/summary.log` (SimplePIR, InsPIRe and VIA, single- and
multi-threaded) and optimistically extrapolates each baseline to the same 
28 settings as step 1, without assuming any memory or computation limits.
It adds the OO-PIR baselines (Piano and RMS), whose primary cost is
bandwidth only, and reads Escape's numbers from the `result/` logs of step 1.
Everything is written to `comparison/NewSummary/`:

- `<entry>/db_log2_<log2N>.log`: the breakdown for one setting (query
  generation, upload, server time, download, total) for OO-PIR, SimplePIR,
  InsPIRe and VIA.
- `bandwidth`: upload + download per query in MiB, for SimplePIR, InsPIRe,
  VIA, Piano, RMS and Escape.
- `compute`: client query generation + server time in seconds for the same
  schemes. Piano and RMS are 0 (bandwidth only). For Escape, memory-stall
  time is excluded.

In `bandwidth` and `compute` the points are grouped by scheme, then by entry
size, one `(index, value)` line per database size, where the index counts
1, 2, … through the sizes in the Table above. The lines can be pasted
directly as plot coordinates.

Other ways to run it (from `comparison/`):

```sh
# throughputs from another summary file, e.g. the previous run (--summary goes before the subcommand)
python3 extrapolate.py --summary summary.prev generate
python3 extrapolate.py --summary summary.prev bandwidth
python3 extrapolate.py --summary summary.prev compute

# write to another folder instead of NewSummary/
python3 extrapolate.py generate  --base-dir OtherSummary
python3 extrapolate.py bandwidth --base-dir OtherSummary
python3 extrapolate.py compute   --base-dir OtherSummary

# only one setting: 64 KiB entries, N = 2^30
python3 extrapolate.py generate --item-size 65536 --log2N 30

# only a range of settings: 16 KiB entries, N = 2^26 ... 2^32
python3 extrapolate.py generate --item-size 16384 --range 26-32
```

## Manual Benchmark

```sh
./bench [n_reps]            # default n_reps = 5
./bench 5 2>&1 | tee bench.log
```

Per rep, the bench prints `query / send / server ans / server cmp / recv
/ recover` timings. After all reps it prints a means summary and `OK` if
every chunk decodes correctly, or `FAIL: <N> error(s)` otherwise.
`bench.log` in this directory is a reference output for the default
parameters.

## Skip-Gen Benchmark (Timing Only, Correctness Fails)

For fast end-to-end timing without paying the AES-CTR cost of generating
each `db_row` / `r_hint`, rebuild with `BENCH_SKIP_GEN=1`. The accumulate
pipeline still runs with the same memory access pattern, so the per-rep
server timings are honest — but the data is zero-filled, so the bench
will print `FAIL: <N> error(s)` at the end instead of `OK`. That failure
is **expected** and indicates the macro is active.

```sh
make EXTRA="-DBENCH_SKIP_GEN=1" -B
./bench 5 2>&1 | tee bench.log.skip
```

To go back to correctness mode, rebuild with `make -B` (or
`-DBENCH_SKIP_GEN=0`).

## Override Parameters (Optional)

```sh
# build the 64 KiB entry / N = 2^30 setting from the paper
make EXTRA="-DBLOCK_KIB=64 -DPLHE_N1=16384 -DPLHE_MR1=256 -DPLHE_MC1=256" -B
./bench 5 2>&1 | tee bench.log.64
```

`-B` forces a rebuild. Any `-D<MACRO>=<value>` in `EXTRA` overrides the
corresponding `params.h` default.

## Other Targets

| target          | purpose                                            |
|-----------------|----------------------------------------------------|
| `make small`    | builds `bench_small` — tiny params, for smoke test |
| `make test_comp`| compress pipeline + Mont modmul micro-bench        |
| `make test_noise`| noise budget sanity check                         |
