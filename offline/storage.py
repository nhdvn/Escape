#!/usr/bin/env python3
"""Latency vs varied storage for the partition shapes in result_/ (made by offline/bench.sh).

For every result_/<block>_30_n<N1>.log it reads
    N1, MR1, MC1      from the parameter line,
    block size        from 'block_size = <bits> bits',
    latency           from the summary line.

A second table gives Piano and RMS at N = 2^30 with 8, 16, 64 KiB entries.
Latency is their per-query bandwidth sent at 70 Mbps.

Writes both tables to storage.log next to this script (and prints them).
"""
import glob
import math
import os
import re

from bandwidth import piano_bandwidth     # Piano / RMS per-query bandwidth formula

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(os.path.dirname(HERE), "result_")
GiB = 1024.0 ** 3

RE_DIMS = re.compile(r"N1=(\d+)\s+MR1=(\d+)\s+MC1=(\d+)")
RE_BLOCK = re.compile(r"block_size\s*=\s*(\d+)\s*bits")
RE_E2E = re.compile(r"^\s*end-to-end\s*=.*=\s*([\d.]+)\s*ms", re.M)
NETWORK_MBPS = 70                                         # latency = bandwidth / 70 Mbps
OOPIR_LOG_N = 30
OOPIR_BLOCKS_KIB = (8, 16, 64)
OOPIR_FACTOR = {"Piano": 40 * 4, "RMS": 40 * 3}


def read_log(path):
    text = open(path).read()
    n1, mr1, mc1 = (int(x) for x in RE_DIMS.search(text).groups())
    block = int(RE_BLOCK.search(text).group(1)) // 8
    m = RE_E2E.search(text)
    assert m, f"no end-to-end summary line in {path}"
    return block, n1, mr1, mc1, float(m.group(1)) / 1000


def main():
    rows = [read_log(p) for p in glob.glob(os.path.join(RESULT, "*_30_n*.log"))]
    rows.sort()                                        # by block, then N1

    hdr = (f"{'block':>7} {'|A| = N1':>9} {'MR1':>6} {'MC1':>6} "
           f"{'latency_s':>10} {'storage_GiB':>12}")
    lines = ["Escape latency vs storage, N = 2^30 entries (from result_/)", "",
             hdr, "-" * len(hdr)]
    for block, n1, mr1, mc1, lat in rows:
        n = n1 * mr1 * mc1
        cells = mr1 * mc1
        alpha = math.log2(n) if cells < math.sqrt(n) else math.log(n)
        storage = cells * alpha * block / GiB
        lines.append(f"{str(block // 1024) + 'KiB':>7} {n1:>9} {mr1:>6} {mc1:>6} "
                     f"{lat:>10.3f} {storage:>12.1f}")

    # OO-PIR: latency = per-query bandwidth (bandwidth.py formula) at NETWORK_MBPS
    n = 1 << OOPIR_LOG_N
    sqrt_n = math.isqrt(n)
    hdr = f"{'block':>7} {'scheme':>7} {'latency_s':>10} {'storage_GiB':>12}"
    lines += ["", f"OO-PIR latency vs storage, N = 2^{OOPIR_LOG_N} entries "
                  f"(latency = bandwidth at {NETWORK_MBPS} Mbps)", "", hdr, "-" * len(hdr)]
    for kib in OOPIR_BLOCKS_KIB:
        block = kib * 1024
        mib = dict(zip(("Piano", "RMS"), piano_bandwidth(n, block)))
        for scheme in ("Piano", "RMS"):
            latency = mib[scheme] * 1024**2 * 8 / (NETWORK_MBPS * 1e6)
            storage = sqrt_n * OOPIR_FACTOR[scheme] * block / GiB
            lines.append(f"{str(kib) + 'KiB':>7} {scheme:>7} {latency:>10.3f} {storage:>12.1f}")

    path = os.path.join(HERE, "storage.log")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
