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

- Linux, x86-64, GCC 9+ with `-march=native` AVX-512 support
- `libgmp-dev` (or distro equivalent), `libgomp` (ships with GCC)
- AVX-512 IFMA52 recommended; the build auto-falls-back to GMP otherwise

Quick check:
```sh
gcc --version            # >= 9
ldconfig -p | grep gmp   # libgmp.so.10 present
grep -o 'avx512ifma' /proc/cpuinfo | head -1   # fast path
```

## Build

```sh
make           # builds ./bench with default params
make clean     # remove build/ and binaries
```

Defaults live in [`plhe/params.h`](plhe/params.h):
`N1 = MR1 = MC1 = 256`, `LAMBDA = 1024`, `BLOCK_KIB = 4`, sigma `3.2`,
which describes a logical 64 GiB database (16 M entries × 4 KiB each).

## Run

```sh
./bench [n_reps]            # default n_reps = 5
./bench 5 2>&1 | tee bench.log
```

Per rep, the bench prints `query / send / server ans / server cmp / recv
/ recover` timings. After all reps it prints a means summary and `OK` if
every chunk decodes correctly, or `FAIL: <N> error(s)` otherwise.
`bench.log` and `bench.log.new` in this directory are reference outputs
for the default parameters.

## Skip-gen benchmark (timing only, correctness fails)

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

## Override parameters (optional)

```sh
# build the 64 KiB entry / N = 2^30 setting from the paper
make EXTRA="-DBLOCK_KIB=64 -DPLHE_N1=16384 -DPLHE_MR1=256 -DPLHE_MC1=256" -B
./bench 5 2>&1 | tee bench.log.64
```

`-B` forces a rebuild. Any `-D<MACRO>=<value>` in `EXTRA` overrides the
corresponding `params.h` default.

## Other targets

| target          | purpose                                            |
|-----------------|----------------------------------------------------|
| `make small`    | builds `bench_small` — tiny params, for smoke test |
| `make test_comp`| compress pipeline + Mont modmul micro-bench        |
| `make test_noise`| noise budget sanity check                         |
