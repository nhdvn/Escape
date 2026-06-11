#!/usr/bin/env python3
"""Append a summary block to each v1 result log under result/{4kb,16kb}/.

For every log file we:
  1. Parse all per-rep stage timings (query comp, client send, answer comp,
     compress, client recv, recover comp).
  2. Drop the first 3 warmup reps; mean the last 5 measured reps.
  3. Append (or replace, idempotently) a summary block:
        bandwidth (send + recv)
        server total delay (answer comp + compress)
        end-to-end delay   (sum of all six stages)

Usage:
    python3 summary.py                  # rewrite under result/{4kb,16kb}/
    python3 summary.py --base-dir DIR   # use a different base
"""
import argparse
import glob
import os
import re

# --- regexes for the per-rep stage lines ------------------------------------
# Each line looks like:
#   [query comp]   85.1 ms
#   [client send]  2.504 MiB  ->  300.1 ms @ 70 Mbps
#   [answer comp]  219.6 ms
#   [compress]     658.3 ms
#   [client recv]  21.500 KiB  ->  2.5 ms @ 70 Mbps
#   [recover comp] 63.2 ms
RE_PARAMS   = re.compile(r'MR1=(\d+)\s+MC1=(\d+)')
RE_QUERY    = re.compile(r'\[query comp\]\s+([\d.]+)\s*ms')
RE_SEND     = re.compile(r'\[client send\]\s+([\d.]+)\s*(MiB|KiB).*?->\s*([\d.]+)\s*ms')
RE_ANSWER   = re.compile(r'\[(?:answer|server)\s+(?:comp|ans)\]\s+([\d.]+)\s*ms')
RE_COMPRESS = re.compile(r'\[(?:compress|server\s+cmp)\]\s+([\d.]+)\s*ms')
RE_RECV     = re.compile(r'\[client recv\]\s+([\d.]+)\s*(MiB|KiB).*?->\s*([\d.]+)\s*ms')
RE_RECOVER  = re.compile(r'\[recover comp\]\s+([\d.]+)\s*ms')
# Per-stage triple: {ans|cmp}_cycle + cpu_stall + mem_stall lines appear
# consecutively under [server ans] / [server cmp]. Anchor on the leading
# *_cycle line to bind cpu_stall / mem_stall unambiguously to one block.
RE_ANS_BLOCK = re.compile(
    r'^\s*ans_cycle\s+([\d.]+)\s*ms\s*\n'
    r'\s*cpu_stall\s+([\d.]+)\s*ms\s*\n'
    r'\s*mem_stall\s+([\d.]+)\s*ms',
    re.MULTILINE)
RE_CMP_BLOCK = re.compile(
    r'^\s*cmp_cycle\s+([\d.]+)\s*ms\s*\n'
    r'\s*cpu_stall\s+([\d.]+)\s*ms\s*\n'
    r'\s*mem_stall\s+([\d.]+)\s*ms',
    re.MULTILINE)

SUMMARY_HEADER = '=== summary'
WARMUPS = 3

def to_kib(v, unit):
    return v * 1024.0 if unit == 'MiB' else v

def parse_log(text):
    pm = RE_PARAMS.search(text)
    mr1 = int(pm.group(1)) if pm else 0
    mc1 = int(pm.group(2)) if pm else 0
    qs   = [float(m.group(1)) for m in RE_QUERY.finditer(text)]
    sds  = [(float(m.group(1)), m.group(2), float(m.group(3))) for m in RE_SEND.finditer(text)]
    ans  = [float(m.group(1)) for m in RE_ANSWER.finditer(text)]
    cmp_ = [float(m.group(1)) for m in RE_COMPRESS.finditer(text)]
    rvs  = [(float(m.group(1)), m.group(2), float(m.group(3))) for m in RE_RECV.finditer(text)]
    rec  = [float(m.group(1)) for m in RE_RECOVER.finditer(text)]
    ans_blocks = [(float(m.group(1)), float(m.group(2)), float(m.group(3)))
                  for m in RE_ANS_BLOCK.finditer(text)]
    cmp_blocks = [(float(m.group(1)), float(m.group(2)), float(m.group(3)))
                  for m in RE_CMP_BLOCK.finditer(text)]
    ans_c    = [b[0] for b in ans_blocks]
    ans_cpus = [b[1] for b in ans_blocks]
    ans_mems = [b[2] for b in ans_blocks]
    cmp_c    = [b[0] for b in cmp_blocks]
    cmp_cpus = [b[1] for b in cmp_blocks]
    cmp_mems = [b[2] for b in cmp_blocks]
    n = min(len(qs), len(sds), len(ans), len(cmp_), len(rvs), len(rec))
    if n == 0:
        return None
    # Keep last (n - WARMUPS) reps, or all if fewer than WARMUPS+1.
    start = WARMUPS if n > WARMUPS else 0
    measured = n - start
    def mean(xs): return sum(xs[start:n]) / measured if xs and len(xs) >= n else 0.0
    return dict(
        n_total=n, n_measured=measured,
        mr1=mr1, mc1=mc1,
        query_ms   = mean(qs),
        send_kib   = sum(to_kib(v, u) for (v, u, _) in sds[start:n]) / measured,
        send_ms    = sum(ms for (_, _, ms) in sds[start:n]) / measured,
        answer_ms  = mean(ans),
        compress_ms= mean(cmp_),
        recv_kib   = sum(to_kib(v, u) for (v, u, _) in rvs[start:n]) / measured,
        recv_ms    = sum(ms for (_, _, ms) in rvs[start:n]) / measured,
        recover_ms = mean(rec),
        ans_cycle  = mean(ans_c),
        ans_cpu    = mean(ans_cpus),
        ans_mem    = mean(ans_mems),
        cmp_cycle  = mean(cmp_c),
        cmp_cpu    = mean(cmp_cpus),
        cmp_mem    = mean(cmp_mems),
    )

def format_summary(s):
    server_ms = s['answer_ms'] + s['compress_ms']
    bw_kib = s['send_kib'] + s['recv_kib']
    bw_str = f"{bw_kib/1024:.3f} MiB" if bw_kib >= 1024 else f"{bw_kib:.2f} KiB"
    send_str = f"{s['send_kib']/1024:.3f} MiB" if s['send_kib'] >= 1024 else f"{s['send_kib']:.2f} KiB"
    recv_str = f"{s['recv_kib']/1024:.3f} MiB" if s['recv_kib'] >= 1024 else f"{s['recv_kib']:.2f} KiB"
    bw_ms = s['send_ms'] + s['recv_ms']
    have_split = (s['ans_cycle'] > 0 or s['ans_cpu'] > 0 or s['ans_mem'] > 0
                  or s['cmp_cycle'] > 0 or s['cmp_cpu'] > 0 or s['cmp_mem'] > 0)
    total_mem = s['ans_mem'] + s['cmp_mem']
    # end-to-end excludes mem_stall (memory wait inside answer + compress),
    # giving the "compute-only" reachable e2e if memory were free.
    e2e_ms = (s['query_ms'] + s['send_ms'] + server_ms - total_mem
              + s['recv_ms'] + s['recover_ms']) if have_split else (
              s['query_ms'] + s['send_ms'] + server_ms + s['recv_ms'] + s['recover_ms'])
    out = (
        f"{SUMMARY_HEADER} (means over {s['n_measured']} measured reps) ===\n"
        f"  bandwidth   = send {send_str} ({s['send_ms']:.1f} ms) + recv {recv_str} ({s['recv_ms']:.1f} ms) = {bw_str} ({bw_ms:.1f} ms)\n"
        f"  srv delay   = answer {s['answer_ms']:.1f} + compress {s['compress_ms']:.1f} = {server_ms:.1f} ms\n"
    )
    if have_split:
        out += (
            f"  ans split   = ans_cycle {s['ans_cycle']:.1f} + cpu_stall {s['ans_cpu']:.1f} "
            f"+ mem_stall {s['ans_mem']:.1f} ms ({100.0 * s['ans_mem'] / s['answer_ms']:.0f}% mem)\n"
        )
        out += (
            f"  cmp split   = cmp_cycle {s['cmp_cycle']:.1f} + cpu_stall {s['cmp_cpu']:.1f} "
            f"+ mem_stall {s['cmp_mem']:.1f} ms ({100.0 * s['cmp_mem'] / s['compress_ms']:.0f}% mem)\n"
        )
        out += (
            f"  end-to-end  = query {s['query_ms']:.1f} + send {s['send_ms']:.1f} + server (no mem_stall) {server_ms - total_mem:.1f} + recv {s['recv_ms']:.1f} + recover {s['recover_ms']:.1f} = {e2e_ms:.1f} ms ({e2e_ms/1000:.2f} s)\n"
        )
    else:
        out += (
            f"  end-to-end  = query {s['query_ms']:.1f} + send {s['send_ms']:.1f} + server {server_ms:.1f} + recv {s['recv_ms']:.1f} + recover {s['recover_ms']:.1f} = {e2e_ms:.1f} ms ({e2e_ms/1000:.2f} s)\n"
        )
    return out

def update_log(path):
    with open(path) as f:
        content = f.read()
    # Strip any prior summary block (idempotent).
    content = re.sub(
        rf'(?ms)^{re.escape(SUMMARY_HEADER)}.*?(?=\n\n|\Z)',
        '', content).rstrip() + '\n'
    s = parse_log(content)
    if s is None:
        print(f"skip (no timings): {path}")
        return
    new = content.rstrip() + '\n\n' + format_summary(s)
    with open(path, 'w') as f:
        f.write(new)
    print(f"updated {path}")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result'))
    p.add_argument('--base-dir', default=default_base,
                   help=f'directory containing 4kb/ 16kb/ 64kb/ subdirs (default: {default_base})')
    args = p.parse_args()
    for sub in ('4kb', '8kb', '16kb', '64kb'):
        for path in sorted(glob.glob(os.path.join(args.base_dir, sub, 'db_log2_*.log'))):
            update_log(path)

if __name__ == '__main__':
    main()
