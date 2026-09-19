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


## Environment

**Hardware**

- x86-64 CPU with AVX-512 IFMA52, AVX-512DQ and AES-NI (e.g. Intel Ice Lake-SP or newer)
- 36 cores recommended (the server uses 36 threads; fewer cores work, only slower)
- 16 GiB of RAM for the default settings; about 128 GiB for the largest ones
  (64 KiB entries with N = 2^30 need about 82 GiB)
- About 1 GiB of disk, and internet access to download the courterparts' codebases

The results in the paper use the AVX-512 IFMA52 code path (`compress/_m512.c`). 
On a CPU without IFMA52 the Makefile still fallback to GMP (`compress/_gmpz.c`), 
but the timings would then not correspond to the paper.

Tested on 2 x Intel Xeon Platinum 8360Y (48 cores in total) with 1 TB of RAM.

**Software**

- Linux (tested on Rocky Linux 8.10)
- GCC with AVX-512 support, GNU Make, GMP and OpenMP (tested with GCC 8.5.0)
- Python 3.8+ (tested with 3.9)
- Go 1.21+ (tested with 1.24), for the SimplePIR baselines
- Rust via rustup, for the InsPIRe baseline (rustup installs the nightly
  toolchain that InsPIRe pins automatically)
- git and g++ with C++17, to download and build the baselines

All versions we tested with are listed in [`metadata.toml`](metadata.toml).


Quick install:
```sh
# GCC, Make, GMP and OpenMP (Debian / Ubuntu)
sudo apt install build-essential libgmp-dev libgomp1

# GCC, Make, GMP and OpenMP (Rocky / RHEL / Fedora)
sudo dnf install gcc gcc-c++ make gmp-devel libgomp

# Rust, via rustup (only needed for the baselines in Steps 2-3)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Go, official release (only needed for the baselines in Steps 2-3)
wget https://go.dev/dl/go1.24.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin             # add to ~/.bashrc to keep it
```

Quick check:
```sh
gcc --version                                   # >= 9
grep -o -w avx512ifma /proc/cpuinfo | head -1   # must print avx512ifma
ldconfig -p | grep gmp                          # libgmp.so.10 present
ldconfig -p | grep gomp                         # libgomp.so.1 present (OpenMP)
python3 --version                               # >= 3.8
go version                                      # >= 1.21
rustup --version && cargo --version             # rustup and cargo present
```

Full hardware and software requirements, including the versions we tested
with, are listed in [`metadata.toml`](metadata.toml).




## Directory Structure

```
bench.c          benchmark driver: builds the virtual DB, runs client/server, times each stage
Makefile         builds ./bench (IFMA52 or GMP compress backend)
metadata.toml    hardware / software requirements and tested versions

plhe/            LWE-based PLHE primitives; params.h holds all parameters (N1, MR1, MC1, BLOCK_KIB, ...)
client/          client: query generation and answer recovery
server/          server: answer computation, then compression of the response
compress/        Paillier compression of the LWE answer
                   _m512.c: AVX-512 IFMA52 backend (used in the paper), mont*.h: Montgomery primitives
                   _gmpz.c: portable GMP fallback
channel/         in-memory client <-> server channel (message exchange)
utils/           virtual database (db.h), perf-counter timing (measure.h), thread barrier,
                   and standalone tests (test_cpu, test_comp, test_noise)

main/            Step 1: bench.sh (28-setting sweep), summary.py (per-log summary),
                   estimator.py (LWE security estimate)
result/          Step 1 output: result/<entry>/db_log2_<log2N>.log

comparison/      Steps 2-3: lattice-based PIR baselines
  measure.py       downloads, builds and measures SimplePIR, InsPIRe, VIA, OSimplePIR
  extrapolate.py   extrapolates the baselines to Escape's settings, writes plot points
  OSimplePIR/      our OpenMP build of SimplePIR
  Measure/         per-run logs of measure.py
  summary.log      measured throughputs (Optimistic section feeds extrapolate.py)
  NewSummary/      extrapolate.py output (per-setting logs, bandwidth, compute, latency)

offline/         Steps 4-5: OO-PIR comparison (Piano, RMS)
  bandwidth.py     per-query and hint-update bandwidth -> bandwidth.log
  bench.sh         Escape runs over partition shapes -> result_/
  storage.py       client storage vs latency -> storage.log
result_/         offline/bench.sh output
```


## Test Build

```sh
make clean     # remove build/ and binaries
make           # builds ./bench with default params
```

Defaults live in [`plhe/params.h`](plhe/params.h):
`N1 = MR1 = MC1 = 256`, `LAMBDA = 1024`, `BLOCK_KIB = 4`, sigma `3.2`,
which describes a logical 64 GiB database (16 M entries × 4 KiB each).

## Reproducing The Paper Results

The comparison in the paper is produced in five steps, run from the
repository root. Step 1 measures Escape, Step 2 measures the baselines, and
Step 3 extrapolates the baselines to Escape's settings and writes the plot
points. Step 4 compares per-query bandwidth with the OO-PIR schemes (Piano,
RMS), and Step 5 compares storage against latency for different partition
shapes. Steps 1-3 must run in order; Steps 4 and 5 are independent of them.

These steps will reproduce the reported numbers/plots in following figures:
- Figure 10: Total Online E2E Latency (After Step 3) 
- Figure 11: Bandwidth Cost with Varied Entry Size (After Step 4)
- Figure 12: Hint-Update Average Bandwidth Per Query (After Step 4)
- Figure 13: Client Storage vs. Latency (Aftet Step 5)

### Step 1: Measure Escape (`main/bench.sh`)

```sh
main/bench.sh                          # writes result/
OUT_BASE=result_rerun main/bench.sh    # or write to another folder
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

### Step 2: Measure Lattice-Based PIR (`comparison/measure.py`)

```sh
cd comparison
python3 measure.py
```

The script downloads the following baselines from GitHub into `comparison/` 
and builds them:

- **SimplePIR** (`ahenzinger/simplepir`)
- **InsPIRe** (`google/private-membership/research/InsPIRe`)
- **VIA** (`owniai/VIA`, which ships its own Intel HEXL library)

It also uses **OSimplePIR**, our OpenMP build of SimplePIR, which is part of
this repository. It then measures each scheme on 2^14, 2^16 and 2^18 entries
of 4 KiB, 5 trials each: SimplePIR, InsPIRe and VIA single-threaded, and
OSimplePIR with 12, 24 and 36 threads. It needs internet access and these
extra tools: git, Go 1.21+, Rust via rustup, and g++ with C++17 and AVX-512.

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
python3 extrapolate.py latency      # plot points: end-to-end latency (s); run after generate
```

The script takes the six throughputs from the Optimistic section of
`comparison/summary.log` (SimplePIR, InsPIRe and VIA, single- and
multi-threaded) and optimistically extrapolates each baseline to the same 
28 settings as Step 1, without assuming any memory or computation limits.
Note the obtained results can witness differences due to extrapolating on
measurement from Step 2. Regardless, the order-of-magnitudes between
Escape and these baselines should remain as reported.

For OO-PIR baselines (Piano and RMS) whose computation cost is negligible,
it computes the query and response bandwidth cost, which is the dominated 
overhead in OO-PIR.

Finally, it reads Escape's numbers from the `result/` logs of Step 1.
Everything is written to `comparison/NewSummary/`:

- `<entry>/db_log2_<log2N>.log`: the breakdown for one setting (query
  generation, upload, server time, download, total) for OO-PIR, SimplePIR,
  InsPIRe and VIA.
- `bandwidth`: upload + download per query in MiB, for SimplePIR, InsPIRe,
  VIA, Piano, RMS and Escape.
- `compute`: client query generation + server time in seconds for the same
  schemes. Piano and RMS are 0 (bandwidth only). For Escape, memory-stall
  time is excluded.
- `latency`: end-to-end latency in seconds (query generation + upload +
  server + download) for the same schemes. It is read from the `total` lines
  of the per-setting logs written by `generate` (so run `generate` first) and
  from the `end-to-end` line of Escape's `result/` logs. For Piano and RMS it
  is the transfer time only; for Escape the server part excludes memory-stall
  time.

In `bandwidth`, `compute` and `latency` the points are grouped by scheme, then by entry
size, one `(index, value)` line per database size, where the index counts
1, 2, … through the sizes in the Table above. The lines can be pasted
directly as plot coordinates.

### Step 4: Compare Bandwidth (`offline/bandwidth.py`)

```sh
python3 offline/bandwidth.py
```

The script computes per-query bandwidth (upload + download, in MiB) from the
formulas of each scheme, with no measurement needed, and writes four tables to
`offline/bandwidth.log`:

- **Table 1**: a fixed 64 TiB database, for entry sizes 2^3 ... 2^18 bytes
  (N = 64 TiB / entry size). Escape vs Piano vs RMS, with Escape's
  N1 x MR1 x MC1 split for each N.
- **Table 2**: a fixed N = 2^30 entries, for the same entry sizes.
- **Tables 3 and 4**: hint-update bandwidth per query for (2^30 x 8 KiB) and
  (2^30 x 64 KiB) databases, for 2^11 ... 2^35 online queries. Each value is
  max(DB / Q, per-query cost): the database download of a hint refresh,
  spread over the Q queries it serves, or the per-query cost of Tables 1 and 2,
  whichever is larger.

- Table 1 and 2 are expected to match with Figure 11.
- Table 3 and 4 are expected to match with Figure 12.

The entry sizes, N and the query counts are constants at the top of the
script (`DEFAULT_EXPS`, `FIXED_LOG_N`, `QUERY_EXPS`).

### Step 5: Compare Storage And Latency (`offline/bench.sh`, `offline/storage.py`)

```sh
offline/bench.sh                           # writes result_/
python3 offline/storage.py                 # writes offline/storage.log
```

`offline/bench.sh` works like `main/bench.sh`, but fixes N = 2^30 and runs 4
partition shapes N1 x MR1 x MC1 for each of the 8, 16 and 64 KiB entry sizes:

| N1 = \|A\| | MR1 = MC1 |
|------------|-----------|
| 1024       | 1024      |
| 4096       | 512       |
| 16384      | 256       |
| 65536      | 128       |

It writes `result_/<entry>_30_n<N1>.log` (skip-gen, 5 repetitions, with the
same summary block as Step 1). Use `OUT_BASE=<dir> offline/bench.sh` to write
elsewhere. The largest setting (64 KiB, N1 = 65536) takes about 15 minutes and
about 82 GiB of RAM.

`offline/storage.py` then writes two tables to `offline/storage.log`:

- **Escape**: for each shape, the end-to-end latency from the `result_/` log
  and the storage MR1 x MC1 x alpha x entry size, where N = N1 x MR1 x MC1 and
  alpha = log2(N) if MR1 x MC1 < sqrt(N), otherwise ln(N).
- **OO-PIR** (Piano, RMS) at N = 2^30 with 8, 16 and 64 KiB entries: the
  latency is the per-query bandwidth of Step 4 sent at 70 Mbps, and the
  storage is sqrt(N) x 40 x 4 x entry size (Piano) or sqrt(N) x 40 x 3 x entry
  size (RMS).

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

## Optional: Other Ways To Run `extrapolate.py`

Run from `comparison/`:

```sh
python3 extrapolate.py --summary summary.prev generate          # throughputs from another summary file
python3 extrapolate.py generate --base-dir OtherSummary         # write to another folder
python3 extrapolate.py generate --item-size 65536 --log2N 30    # one setting: 64 KiB, N = 2^30
python3 extrapolate.py generate --item-size 16384 --range 26-32 # a range: 16 KiB, N = 2^26 ... 2^32
```

`--summary` and `--base-dir` work with every subcommand.

## Other Targets

| target          | purpose                                            |
|-----------------|----------------------------------------------------|
| `make small`    | builds `bench_small` — tiny params, for smoke test |
| `make test_comp`| compress pipeline + Mont modmul micro-bench        |
| `make test_noise`| noise budget sanity check                         |
