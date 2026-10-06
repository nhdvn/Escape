#!/usr/bin/env python3
"""
Compare fresh outputs with the paper's values in plots/ and write rcheck.log.

  NewSummary/{latency,bandwidth,compute}  (Step 3)  vs  plots/{latency,bandwidth,compute}  (Figure 10)
  offline/bandwidth.log                   (Step 4)  vs  plots/bandwidth.log                (Figures 11-12)
  offline/storage.log                     (Step 5)  vs  plots/storage.log                  (Figure 13)

diff = (paper - fresh) / paper; '!' marks |diff| > 10%.

Usage:
    python3 rcheck.py                                  # writes rcheck.log next to this script
    python3 rcheck.py --summary NewSummary_rerun       # another extrapolate.py output dir
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))       # Escape/comparison
REPO = os.path.dirname(HERE)                             # Escape repo root
PLOTS = os.path.join(REPO, "plots")
FLAG = 10.0                                              # percent
SCHEMES = ("SimplePIR", "InsPIRe", "VIA", "Piano", "RMS", "Escape")
METRICS = ("latency", "bandwidth", "compute")            # file names in NewSummary/ and plots/
LOG2N_START = {"4kb": 28, "8kb": 26, "16kb": 26, "64kb": 24}     # extrapolate.py CONFIGS
diffs = []                                               # (|diff| %, label) of every pair


def cell(fresh, paper, label, w):
    """'fresh paper diff' for one value pair, and record the difference."""
    if fresh is None:
        return f"{'-':>{w}} {paper:>{w}g} {'missing':>7}  "
    if fresh == 0:                                       # no model: nothing to compare
        return f"{0:>{w}g} {paper:>{w}g} {'N/A':>7}  "
    d = (paper - fresh) / paper * 100
    diffs.append((abs(d), label))
    return f"{fresh:>{w}g} {paper:>{w}g} {d:>+6.1f}%{' !' if abs(d) > FLAG else '  '}"


def head(prefix, names, w):
    """Two header lines: metric names, then fresh / paper / diff under each."""
    width = 2 * w + 11
    top = " " * len(prefix) + " | " + " | ".join(f"{n:^{width}}" for n in names)
    sub = prefix + " | " + " | ".join(f"{'fresh':>{w}} {'paper':>{w}} {'diff':>7}  " for _ in names)
    return [top, sub, "-" * len(sub)]


def rel(path):
    return os.path.relpath(path, REPO)


def read_points(path):
    """{(scheme, block): [values]} from a NewSummary-style file."""
    out, scheme, block = {}, None, None
    for line in open(path) if os.path.exists(path) else []:
        m = re.match(r"=== (\w+) ===", line)
        if m:
            scheme, block = m.group(1), None
            continue
        m = re.match(r"\s+(\d+kb):", line)
        if m:
            block = m.group(1)
            out[(scheme, block)] = []
            continue
        m = re.match(r"\s+\(\d+, ([\d.]+)\)", line)
        if m and block:
            out[(scheme, block)].append(float(m.group(1)))
    return out


def read_tables(path):
    """{k: (title, {row key: {column: value}})} for every 'Table k:' block of a bandwidth.log."""
    tables, rows, cols = {}, None, None
    for line in open(path) if os.path.exists(path) else []:
        m = re.match(r"Table (\d+):", line)
        tok = line.split()
        if m:
            rows, cols = {}, None
            tables[int(m.group(1))] = (line.strip(), rows)
        elif rows is None or not tok or set(line.strip()) <= set("-"):
            continue
        elif cols is None:
            cols = tok if tok[0] in ("2^x", "Q") else None
        elif len(tok) == len(cols) and tok[0].startswith("2^"):
            rows[tok[0]] = dict(zip(cols[1:], map(float, tok[1:])))
    return tables


def read_storage(path):
    """{(block, scheme, |A|): (latency_s, storage_GiB)} from a storage.log."""
    out = {}
    for line in open(path) if os.path.exists(path) else []:
        tok = line.split()
        if len(tok) == 6 and tok[0].endswith("KiB"):            # Escape: block N1 MR1 MC1 lat st
            out[(tok[0], "Escape", tok[1])] = (float(tok[4]), float(tok[5]))
        elif len(tok) == 4 and tok[0].endswith("KiB"):          # OO-PIR: block scheme lat st
            out[(tok[0], tok[1], "")] = (float(tok[2]), float(tok[3]))
    return out


def online(summary):
    fresh = {k: read_points(os.path.join(summary, k)) for k in METRICS}
    paper = {k: read_points(os.path.join(PLOTS, k)) for k in METRICS}
    blocks = sorted({b for _, b in paper["latency"]}, key=lambda b: int(b[:-2]))
    lines = [f"[1] Online cost per scheme: {rel(summary)}/{{latency,bandwidth,compute}} vs "
             f"plots/ (Figure 10)", ""]
    lines += head(f"{'block':>5} {'log2N':>5} {'scheme':>9}",
                  ("latency (s)", "bandwidth (MiB)", "compute (s)"), 9)
    for block in blocks:
        for scheme in SCHEMES:
            pts = {k: paper[k].get((scheme, block), []) for k in METRICS}
            for i, _ in enumerate(pts["latency"]):
                log2n = LOG2N_START[block] + i
                got = {k: fresh[k].get((scheme, block), []) for k in METRICS}
                lines.append(f"{block:>5} {log2n:>5} {scheme:>9} | " + " | ".join(
                    cell(got[k][i] if i < len(got[k]) else None, pts[k][i],
                         f"{k} {scheme} {block} 2^{log2n}", 9) for k in METRICS))
            lines.append("")
    return lines


def bandwidth(offline):
    fresh = read_tables(os.path.join(offline, "bandwidth.log"))
    paper = read_tables(os.path.join(PLOTS, "bandwidth.log"))
    lines = [f"[2] Per-query bandwidth (MiB): {rel(offline)}/bandwidth.log vs plots/bandwidth.log "
             f"(Figures 11-12)"]
    for k, (title, rows) in sorted(paper.items()):
        got = fresh.get(k, ("", {}))[1]
        lines += ["", title, ""] + head(f"{'':>5}", ("Escape", "Piano", "RMS"), 9)
        for key, cols in rows.items():
            lines.append(f"{key:>5} | " + " | ".join(
                cell(got.get(key, {}).get(s), cols[s], f"bandwidth table {k} {s} {key}", 9)
                for s in ("Escape", "Piano", "RMS")))
    return lines


def storage(offline):
    fresh = read_storage(os.path.join(offline, "storage.log"))
    paper = read_storage(os.path.join(PLOTS, "storage.log"))
    lines = [f"[3] Client storage vs latency: {rel(offline)}/storage.log vs plots/storage.log "
             f"(Figure 13)", ""]
    lines += head(f"{'block':>6} {'scheme':>6} {'|A|':>6}", ("latency (s)", "storage (GiB)"), 8)
    order = {"Escape": 0, "Piano": 1, "RMS": 2}
    for key in sorted(paper, key=lambda k: (int(k[0][:-3]), order[k[1]], int(k[2] or 0))):
        (lat, st), got = paper[key], fresh.get(key, (None, None))
        label = f"{key[0]} {key[1]}" + (f" |A|={key[2]}" if key[2] else "")
        lines.append(f"{key[0]:>6} {key[1]:>6} {key[2]:>6} | "
                     + cell(got[0], lat, f"storage latency {label}", 8) + " | "
                     + cell(got[1], st, f"storage size {label}", 8))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default=os.path.join(HERE, "NewSummary"),
                   help="extrapolate.py output dir (Step 3)")
    p.add_argument("--offline", default=os.path.join(REPO, "offline"), help="Steps 4-5 log dir")
    p.add_argument("--out", default=os.path.join(HERE, "rcheck.log"))
    a = p.parse_args()

    lines = ["rcheck: fresh outputs vs the paper's values (plots/)",
             f"diff = (paper - fresh) / paper;  ! = |diff| > {FLAG:g}%;  N/A = no compute model "
             f"(Piano, RMS)", ""]
    lines += online(os.path.abspath(a.summary)) + [""]
    lines += bandwidth(os.path.abspath(a.offline)) + [""]
    lines += storage(os.path.abspath(a.offline)) + [""]

    within = lambda pct: sum(d <= pct for d, _ in diffs)
    lines += [f"Summary: {len(diffs)} values compared; {within(1)} within 1%, {within(5)} within 5%, "
              f"{within(FLAG)} within {FLAG:g}%"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
