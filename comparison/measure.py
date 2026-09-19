#!/usr/bin/env python3
"""Download SimplePIR, InsPIRe and VIA, run each on 2^LOG_N records of 4 KiB for
several LOG_N, and write each scheme's server throughput to summary.log.

  throughput = database size / server answer time   (mean over 5 trials)

Sources (SimplePIR/, InsPIRe/, VIA/) go next to this script, per-run logs go to
Measure/ (emptied at the start of every run), and summary.log stays next to this script
(the previous summary.log is moved to summary.prev before each run).
SimplePIR, InsPIRe and VIA run single-threaded. OSimplePIR (our OpenMP build of
SimplePIR, must already be in this folder) is swept over 12, 24, 36 threads.
Needs: git, go, cargo (rustup), g++.
Usage: python3 measure.py
"""
import json, os, re, shutil, subprocess

LOG_NS, ITEM_BYTES = (14, 16, 18), 4096            # database sizes 2^LOG_N x 4 KiB
THREADS = (12, 24, 36)                             # OSimplePIR thread sweep
WORK = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(WORK, "Measure")
os.environ["PATH"] += ":" + os.path.expanduser("~/.cargo/bin")
os.environ["RAYON_NUM_THREADS"] = "1"              # keep InsPIRe single-threaded too


def sh(cmd, cwd=WORK, log=None, stdin=None):
    """Run a shell command (optionally capturing output to a log); stop on error."""
    print("$", cmd, flush=True)
    out = open(log, "w") if log else None
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, text=True, input=stdin,
                   stdout=out, stderr=subprocess.STDOUT if out else None)


def clone(url, name):
    if not os.path.isdir(os.path.join(WORK, name)):
        sh(f"git clone --depth 1 {url} {name}")


def mean(xs):
    assert xs, "no timings found in log"
    return sum(xs) / len(xs)


def go_seconds(s):
    """Go duration string ('220.5µs', '1.99ms', '1.2s') -> seconds."""
    for unit, f in (("ns", 1e-9), ("µs", 1e-6), ("us", 1e-6), ("ms", 1e-3), ("s", 1)):
        if s.endswith(unit):
            return float(s[:-len(unit)]) * f


def simplepir(log_n, folder="SimplePIR", env="", tag=None):
    if folder == "SimplePIR":                      # OSimplePIR is expected to exist already
        clone("https://github.com/ahenzinger/simplepir", "SimplePIR")
    log = f"{LOGS}/{tag or folder.lower()}_n{log_n}.log"
    sh(f"{env} LOG_N={log_n} D={ITEM_BYTES * 8} go test -bench=BenchmarkSimplePirSingle -run='^$' -timeout 0",
       cwd=f"{WORK}/{folder}/pir", log=log)
    # each trial prints "Answering query...\n\tElapsed: <time>"
    t = mean([go_seconds(x) for x in re.findall(r"Answering query...\s+Elapsed: (\S+)", open(log).read())])
    return t, t                                    # the whole answer is one pass over the DB


def inspire(log_n):
    d = f"{WORK}/InsPIRe"
    if not os.path.isdir(d):   # InsPIRe is a subfolder of google/private-membership: keep only it
        sh("git clone --depth 1 --filter=blob:none --sparse https://github.com/google/private-membership pm")
        sh("git sparse-checkout set research/InsPIRe", cwd=f"{WORK}/pm")
        sh("mv pm/research/InsPIRe InsPIRe && rm -rf pm")
    sh("cargo build --release --bin inspire", cwd=d)
    js = f"{LOGS}/inspire_n{log_n}.json"
    sh(f"./target/release/inspire --num-items {1 << log_n} --item-size-bits {ITEM_BYTES * 8} "
       f"--dim0 {1 << (log_n // 2 + 5)} --trials 5 --out-report-json {js}",
       cwd=d, log=f"{LOGS}/inspire_n{log_n}.log")
    on = json.load(open(js))["online"]
    return on["serverTimeMs"] / 1e3, on["firstPassTimeUs"] / 1e6


def via(log_n):
    clone("https://github.com/owniai/VIA", "VIA")  # ships HEXL headers + prebuilt lib/libhexl.a
    v = f"{WORK}/VIA"

    # VIA stores 4 KiB records; N = 2^(LOG_ROW + LOG_COL - 1), split as in our earlier VIA runs
    log_row = log_n // 2 - 2
    core = f"{v}/src/core.hpp"
    src = re.sub(r"LOG_ROW = \d+;", f"LOG_ROW = {log_row};", open(core).read())
    src = re.sub(r"LOG_COL = \d+;", f"LOG_COL = {log_n + 1 - log_row};", src)
    open(core, "w").write(src)

    # the bundled libhexl.a needs __libc_single_threaded, missing from glibc < 2.32 (e.g. RHEL 8)
    sh("echo 'char __libc_single_threaded;' > stub.c && gcc -c stub.c -o stub.o "
       "&& g++ -std=c++17 -O3 -DNDEBUG -march=native -mavx512dq -mavx512ifma "
       "-Iinclude src/*.cpp lib/libhexl.a stub.o -o VIA_PIR", cwd=v)
    log = f"{LOGS}/via_n{log_n}.log"
    sh("./VIA_PIR", cwd=v, log=log, stdin="0 0 5\n")   # VIA, no blinded extraction, 5 trials
    text = open(log).read()
    total = mean([float(x) for x in re.findall(r"Total Answer Time: ([\d.]+) ms", text)])
    first = mean([float(x) for x in re.findall(r"First Dimension: ([\d.]+) ms", text)])
    return total / 1e3, first / 1e3


def append_optimistic(path):
    """Append an 'optimistic' first-pass throughput per scheme to the summary: the mean
    of its 3 highest rows, pooled over all sizes (and all OSimplePIR thread counts).
    InsPIRe and VIA are then scaled by the OSimplePIR / SimplePIR ratio to estimate
    multi-threaded OInsPIRe / OVIA."""
    text = open(path).read().split("\nOptimistic")[0].rstrip("\n")   # drop an older section
    fp = {}
    for row in text.splitlines()[1:]:              # columns: ... scheme threads ... first-pass
        f = row.split()
        fp.setdefault(f[2], []).append(float(f[6]))

    opt = {k: mean(sorted(v)[-3:]) for k, v in fp.items()}     # mean of top 3
    ratio = opt["OSimplePIR"] / opt["SimplePIR"]
    lines = [text, "",
             "Optimistic first-pass throughput (GiB/s) = mean of top 3 (across all thread counts)",
             f"  OSimplePIR {opt['OSimplePIR']:7.2f}",
             f"  SimplePIR  {opt['SimplePIR']:7.2f}",
             f"  ratio OSimplePIR / SimplePIR = {ratio:.2f}",
             f"  InsPIRe    {opt['InsPIRe']:7.2f}   -> OInsPIRe = InsPIRe x ratio = {opt['InsPIRe'] * ratio:.2f}",
             f"  VIA        {opt['VIA']:7.2f}   -> OVIA     = VIA x ratio     = {opt['VIA'] * ratio:.2f}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    if os.path.exists(f"{WORK}/summary.log"):      # keep the previous run's summary
        os.replace(f"{WORK}/summary.log", f"{WORK}/summary.prev")
    shutil.rmtree(LOGS, ignore_errors=True)       # start every run with an empty Measure/
    os.makedirs(LOGS)
    lines = [f"{'log2N':>5} {'DB_MiB':>7} {'scheme':10} {'threads':>7} {'server_s':>9} "
             f"{'server GiB/s':>13} {'first-pass GiB/s':>17}"]
    for log_n in LOG_NS:
        results = []                               # (scheme, threads, (server_s, first_s))
        for t in THREADS:                          # OSimplePIR thread sweep, not pinned
            results.append(("OSimplePIR", t, simplepir(log_n, "OSimplePIR", f"OMP_NUM_THREADS={t}",
                                                       f"osimplepir_{t}t")))
        results += [("SimplePIR", 1, simplepir(log_n)),
                    ("InsPIRe",   1, inspire(log_n)),
                    ("VIA",       1, via(log_n))]

        db_gib = (1 << log_n) * ITEM_BYTES / 2**30
        for name, threads, (server_s, first_s) in results:
            lines.append(f"{log_n:5} {db_gib * 1024:7.0f} {name:10} {threads:7} {server_s:9.4f} "
                         f"{db_gib / server_s:13.2f} {db_gib / first_s:17.2f}")
        summary = "\n".join(lines) + "\n"
        open(f"{WORK}/summary.log", "w").write(summary)   # rewritten after each size
        print(summary, flush=True)
    append_optimistic(f"{WORK}/summary.log")


if __name__ == "__main__":
    main()
