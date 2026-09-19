#!/usr/bin/env python3
"""
Extrapolate SimplePIR / InsPIRe / VIA online delay (plus OO-PIR bandwidth).

Subcommands:
  generate   Write OO-PIR + SimplePIR + InsPIRe + VIA extrapolation summary files into
             NewSummary/{4kb,8kb,16kb,64kb}/db_log2_*.log (next to this script).
  bandwidth  Write NewSummary/bandwidth (upload + download per scheme).
  compute    Write NewSummary/compute (client_qgen + server per scheme).
  latency    Write NewSummary/latency (end-to-end latency per scheme, from the
             generate logs and Escape's logs; run generate first).

Usage:
    python3 extrapolate.py generate                                          # all scales (4/8/16/64 KiB)
    python3 extrapolate.py generate --item-size 4096 --log2N 28              # one cell, 4 KiB block
    python3 extrapolate.py generate --item-size 16384 --range 26-32          # 16 KiB sweep
    python3 extrapolate.py --summary summary.prev generate                   # rates from the previous summary
"""
import argparse
import os
import re
import math

# First-pass throughput (GiB/s), read from the "Optimistic" section that measure.py
# appends to summary.log next to this script (multi = O* estimate, single = measured).
HERE = os.path.dirname(os.path.abspath(__file__))       # Escape/comparison
REPO = os.path.dirname(HERE)                             # Escape repo root
SUMMARY_LOG = os.path.join(HERE, "summary.log")

def _read_rates(path):
    text = open(path).read()
    assert "Optimistic" in text, f"no 'Optimistic' section in {path} (run measure.py first)"
    text = text[text.index("Optimistic"):]
    def num(pattern):
        m = re.search(pattern, text)
        assert m, f"'{pattern}' not found in {path}"
        return float(m.group(1))
    return {
        "sp_multi":   num(r"\n\s*OSimplePIR\s+([\d.]+)"),
        "sp_single":  num(r"\n\s*SimplePIR\s+([\d.]+)"),
        "ip_multi":   num(r"OInsPIRe\s*=.*?=\s*([\d.]+)"),
        "ip_single":  num(r"\n\s*InsPIRe\s+([\d.]+)"),
        "via_multi":  num(r"OVIA\s*=.*?=\s*([\d.]+)"),
        "via_single": num(r"\n\s*VIA\s+([\d.]+)"),
    }

def load_rates(path=SUMMARY_LOG):
    """Set the six first-pass rates below from a measure.py summary file."""
    global SIMPLEPIR_RATE_MULTI_GIBS, SIMPLEPIR_RATE_SINGLE_GIBS, INSPIRE_RATE_MULTI_GIBS
    global INSPIRE_RATE_SINGLE_GIBS, VIA_RATE_MULTI_GIBS, VIA_RATE_SINGLE_GIBS
    r = _read_rates(path)
    SIMPLEPIR_RATE_MULTI_GIBS  = r["sp_multi"]     # OSimplePIR (OpenMP), multi-thread
    SIMPLEPIR_RATE_SINGLE_GIBS = r["sp_single"]    # SimplePIR, single thread
    INSPIRE_RATE_MULTI_GIBS    = r["ip_multi"]     # OInsPIRe = InsPIRe x OSimplePIR/SimplePIR
    INSPIRE_RATE_SINGLE_GIBS   = r["ip_single"]    # InsPIRe, single thread
    VIA_RATE_MULTI_GIBS        = r["via_multi"]    # OVIA = VIA x OSimplePIR/SimplePIR
    VIA_RATE_SINGLE_GIBS       = r["via_single"]   # VIA FirstDim, single thread

NETWORK_MBPS = 70
def net_ms(b): return b * 8 / (NETWORK_MBPS * 1000.0)

# ── Common LWE/RLWE parameters ────────────────────────────────────────────────
POLY_LEN = 2048
LOG_Q_BYTES = 7        # ⌈log_q/8⌉ = 56/8

# ──────────────────────────────────────────────────────────────────────────────
# OO-PIR sqrt-bandwidth tradeoff (top preamble of each NewSummary log)
# ──────────────────────────────────────────────────────────────────────────────
def oopir_estimate(log2N, item_size_bits):
    """OO-PIR sqrt-bandwidth analysis:
         (1) Piano download       : sqrtN · block bytes
         (2) RMS download         : 2·sqrtN/log(sqrtN) · block bytes
         (3) shared query upload  : sqrtN · log(sqrtN) bits
       total Piano = (1) + (3); total RMS = (2) + (3). Wire-time at NETWORK_MBPS."""
    import math
    N = 1 << log2N
    sqrtN = round(math.sqrt(N))
    log_sqrtN = (log2N + 1) // 2
    block_bytes = item_size_bits // 8

    piano_bytes = sqrtN * block_bytes
    rms_bytes = 2 * (sqrtN // log_sqrtN) * block_bytes
    query_bits  = sqrtN * log_sqrtN
    query_bytes = query_bits / 8
    off_bytes = N * block_bytes        # full-DB download (offline)

    piano_ms = net_ms(piano_bytes)
    rms_ms = net_ms(rms_bytes)
    query_ms = net_ms(query_bytes)
    ms_off = net_ms(off_bytes)
    return dict(
        N=N, sqrtN=sqrtN, log_sqrtN=log_sqrtN, block_bytes=block_bytes,
        piano_bytes=piano_bytes, piano_mib=piano_bytes / 1024**2, piano_ms=piano_ms,
        rms_bytes=rms_bytes, rms_mib=rms_bytes / 1024**2, rms_ms=rms_ms,
        query_bits=query_bits,   query_kib=query_bytes / 1024,    query_ms=query_ms,
        off_bytes=off_bytes, off_gib=off_bytes / 1024**3, ms_off=ms_off,
        total_piano_ms=piano_ms + query_ms, total_rms_ms=rms_ms + query_ms,
    )

def format_oopir_block(e):
    return (
        f"OO-PIR sqrt-bandwidth @ {NETWORK_MBPS} Mbps:\n"
        f"  (1) √N · block       = {e['sqrtN']} · {e['block_bytes']} B = {e['piano_mib']:.2f} MiB → {e['piano_ms']:.1f} ms\n"
        f"  (2) 2·√N / log(√N) · block = {2*(e['sqrtN']//e['log_sqrtN'])} · {e['block_bytes']} B = {e['rms_mib']:.2f} MiB → {e['rms_ms']:.1f} ms\n"
        f"  (3) √N · log(√N)     = {e['sqrtN']} · {e['log_sqrtN']} bits = {e['query_kib']:.1f} KiB → {e['query_ms']:.1f} ms\n"
        f"  total PIANO = (1) + (3) = {e['total_piano_ms']/1000:.2f} s\n"
        f"  total RMS   = (2) + (3) = {e['total_rms_ms']/1000:.2f} s\n"
    )

# ──────────────────────────────────────────────────────────────────────────────
# SimplePIR model (Henzinger et al. USENIX'23, with OpenMP build of pir.c)
# ──────────────────────────────────────────────────────────────────────────────
# SimplePIR (Henzinger et al.) real square layout. logq=32 (params.csv) -> 4 B/coeff.
SP_LOG_Q_BYTES = 4
SP_QGEN_US_PER_SAMPLE = 1.04      # measured: "Building query" 261 ms / m=250407 (4kb n=2^24 log)

def _sp_logp(logm):
    """Plaintext-modulus bits vs log2(M), fit to params.csv (logn=10, logq=32):
       log2(p) = 9,9,9,9,8,8,8,8,7 for logm=13..21 -> ~9.953 - 0.25*(logm-13)."""
    return max(2.0, 9.953 - 0.25 * (logm - 13.0))

def simplepir_estimate(log2N, item_size_bits, single_thread=False):
    """Real SimplePIR (square, entry-level). database.go: each B-byte record is
       Ne = ceil(B_bits/log2 p) Z_p digits; db_elems = N*Ne is reshaped to L x M
       with L ~= M ~= sqrt(db_elems) (ApproxSquareDatabaseDims). simple_pir.go:
       online upload = M*logq/8 (query is the M-vector A*s+e), online download =
       L*logq/8 (answer is the selected L-column). Server still scans the full DB."""
    N = 1 << log2N
    item_bytes = item_size_bits // 8
    db_bytes = N * item_bytes

    # Resolve (log p, M, L) self-consistently: M sets logm, logm sets p, p sets Ne.
    # L,M from ApproxSquareDatabaseDims (database.go): l=floor(sqrt(db_elems))
    # rounded up to a multiple of Ne, then m=ceil(db_elems/l).
    def dims(logp):
        ne = math.ceil(item_size_bits / logp)
        db_elems = N * ne
        l = math.floor(math.sqrt(db_elems))
        rem = l % ne
        if rem:
            l += ne - rem
        m = math.ceil(db_elems / l)
        return ne, db_elems, l, m
    logp = 8.0
    for _ in range(30):
        _, _, _, m = dims(logp)
        new_logp = _sp_logp(math.log2(m))
        if abs(new_logp - logp) < 1e-4:
            break
        logp = new_logp
    ne, db_elems, l, m = dims(logp)            # m = DB width (query), l = DB height (answer)

    rate = SIMPLEPIR_RATE_SINGLE_GIBS if single_thread else SIMPLEPIR_RATE_MULTI_GIBS
    first_pass_ms = round(db_bytes / (rate * 1024**3) * 1000)
    server_ms = first_pass_ms                  # O(N*B) plaintext matmul over the packed DB

    upload_b = m * SP_LOG_Q_BYTES              # query = M LWE coefficients
    download_b = l * SP_LOG_Q_BYTES            # answer = L LWE coefficients
    upload_ms = round(net_ms(upload_b))
    download_ms = round(net_ms(download_b))

    qgen_ms = round(SP_QGEN_US_PER_SAMPLE * m / 1000)   # M LWE encryptions

    total_ms = qgen_ms + upload_ms + server_ms + download_ms
    return dict(
        scheme='SimplePIR', log2N=log2N, item_kib=item_bytes // 1024,
        db_phys_tib=db_bytes / 1024**4, padding=1.0,
        upload_b=upload_b, upload_mib=upload_b / 1024**2, upload_ms=upload_ms,
        download_b=download_b, download_mib=download_b / 1024**2, download_ms=download_ms,
        qgen_ms=qgen_ms, first_pass_ms=first_pass_ms, pack_ms=0, rgsw_ms=0,
        server_ms=server_ms, total_ms=total_ms, total_s=total_ms / 1000,
        sp_m=m, sp_l=l, sp_ne=ne, sp_logp=logp,
    )

# ──────────────────────────────────────────────────────────────────────────────
# InsPIRe model (bin/inspire.rs:772-829)
# ──────────────────────────────────────────────────────────────────────────────
INSPIRE_LOG_P = 16
INSPIRE_T_MAX = 32
INSPIRE_PUB_PARAMS_B = 86016                  # InspiRING: 2 KS matrices, 84 KiB
INSPIRE_RGSW_QUERY_B = 86016                  # 1 RGSW poly-eval point, 84 KiB
INSPIRE_RLWE_DOWNLOAD_B = 12288
INSPIRE_PACK_MS_PER_INST_ST = 17              # single-thread, fitted (1067ms/64 instances at log2N=22)
INSPIRE_RGSW_MS_PER_C_ST = 7                  # single-thread, fitted (14ms/2 c at log2N=22)
INSPIRE_QGEN_RATE_MS_PER_DIM0 = 0.0262        # single-thread, fitted

def heuristic_dim0(log2N):
    return 1 << ((log2N + 1) // 2 + 5)

def inspire_estimate(log2N, item_size_bits, single_thread=False):
    N = 1 << log2N
    item_bytes = item_size_bits // 8
    cells = item_size_bits // INSPIRE_LOG_P
    total_cells = N * cells
    dim0 = heuristic_dim0(log2N)
    db_cols_cells = max(POLY_LEN, total_cells // dim0)
    db_cols_poly = max(1, db_cols_cells // POLY_LEN)
    # Interpolate degree grows up to T_MAX
    t_target = min(INSPIRE_T_MAX, max(1, db_cols_poly))
    # Real code: t doubles until ≥ dim1_lower_bound, capped at T_MAX
    t = 1
    while t < t_target and t * 2 <= INSPIRE_T_MAX:
        t *= 2
    c = max(1, db_cols_poly // t)

    # Upload
    query_indicator_b = dim0 * LOG_Q_BYTES
    upload_b = query_indicator_b + INSPIRE_PUB_PARAMS_B + INSPIRE_RGSW_QUERY_B
    upload_ms = round(net_ms(upload_b))

    # Download
    download_b = c * INSPIRE_RLWE_DOWNLOAD_B
    download_ms = round(net_ms(download_b))

    # Server
    db_bytes = N * item_bytes                  # log_p=16 → no padding
    rate = INSPIRE_RATE_SINGLE_GIBS if single_thread else INSPIRE_RATE_MULTI_GIBS
    first_pass_ms = round(db_bytes / (rate * 1024**3) * 1000)

    pack_st_ms = INSPIRE_PACK_MS_PER_INST_ST * db_cols_poly
    rgsw_st_ms = INSPIRE_RGSW_MS_PER_C_ST * c
    if single_thread:
        pack_ms = pack_st_ms
        rgsw_ms = rgsw_st_ms
    else:
        pack_ms = max(1, round(pack_st_ms / 24))
        rgsw_ms = max(1, round(rgsw_st_ms / 24))
    server_ms = first_pass_ms + pack_ms + rgsw_ms

    qgen_ms = round(INSPIRE_QGEN_RATE_MS_PER_DIM0 * dim0)

    total_ms = qgen_ms + upload_ms + server_ms + download_ms
    return dict(
        scheme='InsPIRe', log2N=log2N, item_kib=item_bytes // 1024,
        dim0=dim0, t=t, db_cols_poly=db_cols_poly, c=c, padding=1.0,
        db_phys_tib=db_bytes / 1024**4,
        upload_b=upload_b, upload_mib=upload_b / 1024**2, upload_ms=upload_ms,
        download_b=download_b, download_kib=download_b / 1024, download_ms=download_ms,
        qgen_ms=qgen_ms, first_pass_ms=first_pass_ms, pack_ms=pack_ms, rgsw_ms=rgsw_ms,
        server_ms=server_ms, total_ms=total_ms, total_s=total_ms / 1000,
    )

# ──────────────────────────────────────────────────────────────────────────────
# VIA model (eprint 2025/2074, src/functions.cpp:326-327, core.hpp)
# ──────────────────────────────────────────────────────────────────────────────
VIA_NATIVE_CHUNK_BYTES = 4096                 # 8 RLWE cts × DEGREE2(512) × log_p(8)/8
VIA_CELL_BYTES = 2048                         # DEGREE1(2048) × log_p(8)/8 plaintext per cell
VIA_DOWNLOAD_KIB_PER_QUERY = 23.0             # 8·(sizePolyDeg2Q3 + sizePolyDeg2Q4)
VIA_QGEN_MS_PER_QUERY = 7                     # client Query() time, ~7 ms at log2N=20
VIA_OVERHEAD_FRAC = 0.05                      # non-FirstDim fraction at large N

def via_split_lr_lc(lr_plus_lc):
    """Pick (LR, LC) to minimize upload at given LR+LC.
       Upload coeff: 57·LR + 15.3·LC, so minimize LR subject to LR ≥ 3 and LC ≥ 3."""
    lr = max(3, lr_plus_lc // 4)              # paper's 32 GB run used LR≈6, LC≈27 (ratio ~1:4)
    lc = max(3, lr_plus_lc - lr)
    return lr, lc

def via_upload_kib_per_query(lr, lc):
    """src/functions.cpp:326. Returns KiB."""
    return (4 * (lr - 3) + 8) * 14.25 + 4 * 8.75 + 7 * (lc - 3) * 2.1875

def via_estimate(log2N, item_size_bits, single_thread=False):
    """VIA at N records of B bytes. Multi-sub-DB layout with SELECTOR REUSE:
       split the dataset into ⌈B/4 KiB⌉ shape-identical sub-DBs (each of N
       records × 4 KiB), with chunk-i of item-k living at the SAME (row,
       column-group) position in every sub-DB.

       The client sends ONE selector; the server applies it to all sub-DBs
       and returns one answer per sub-DB.

       Upload:    1 selector (≈fixed, scales only with log N — sized for
                  sub-DB cell count log2N+1, NOT with item size).
       Download:  ⌈B/4 KiB⌉ × 23 KiB (scales linearly with item size — this
                  is the inherent ciphertext expansion).
       Server:    same total FirstDim work as scanning the full DB
                  (item_bytes · N), regardless of layout."""
    N = 1 << log2N
    item_bytes = item_size_bits // 8
    native_queries = max(1, item_bytes // VIA_NATIVE_CHUNK_BYTES)

    # Per-sub-DB cell count: each sub-DB has N records of 4 KiB = 2N cells
    lr_plus_lc = log2N + 1
    lr, lc = via_split_lr_lc(lr_plus_lc)

    # ONE selector, reused across all sub-DBs
    upload_kib   = via_upload_kib_per_query(lr, lc)
    download_kib = native_queries * VIA_DOWNLOAD_KIB_PER_QUERY
    upload_b = round(upload_kib * 1024)
    download_b = round(download_kib * 1024)
    upload_ms = round(net_ms(upload_b))
    download_ms = round(net_ms(download_b))

    # Server: total FirstDim work scales with full DB (= sum over sub-DBs)
    full_db_bytes = N * item_bytes
    rate = VIA_RATE_SINGLE_GIBS if single_thread else VIA_RATE_MULTI_GIBS
    first_pass_ms = full_db_bytes / (rate * 1024**3) * 1000
    server_ms = round(first_pass_ms * (1 + VIA_OVERHEAD_FRAC))
    first_pass_ms = round(first_pass_ms)

    # Client query gen: ONE selector regardless of native_queries (selector reused)
    qgen_ms = VIA_QGEN_MS_PER_QUERY

    total_ms = qgen_ms + upload_ms + server_ms + download_ms
    return dict(
        scheme='VIA', log2N=log2N, item_kib=item_bytes // 1024,
        native_queries=native_queries, lr=lr, lc=lc,
        db_phys_tib=(N * item_bytes) / 1024**4, padding=1.0,
        upload_b=upload_b, upload_mib=upload_b / 1024**2, upload_ms=upload_ms,
        download_b=download_b, download_kib=download_b / 1024, download_ms=download_ms,
        qgen_ms=qgen_ms, first_pass_ms=first_pass_ms, pack_ms=0, rgsw_ms=0,
        server_ms=server_ms, total_ms=total_ms, total_s=total_ms / 1000,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────────────
ESTIMATORS = [simplepir_estimate, inspire_estimate, via_estimate]

def format_simplepir_block(e):
    return (
        f"SimplePIR-style total online delay (≥, 70 Mbps; "
        f"first_pass at {e.get('rate', SIMPLEPIR_RATE_MULTI_GIBS):.1f} GiB/s; "
        f"real square layout M={e.get('sp_m','?')} L={e.get('sp_l','?')} Ne={e.get('sp_ne','?')} log2p~{e.get('sp_logp',0):.1f}):\n"
        f"  client_qgen={e['qgen_ms']}ms  "
        f"upload={e['upload_mib']:.2f} MiB→{e['upload_ms']}ms  "
        f"server={e['server_ms']}ms (first_pass={e['first_pass_ms']})  "
        f"download={e['download_mib']:.2f} MiB→{e['download_ms']}ms  decode=0\n"
        f"  total = {e['total_ms']} ms ({e['total_s']:.1f} s)\n"
    )

def format_inspire_block(e):
    return (
        f"InsPIRe-style total online delay (≥, 70 Mbps; "
        f"first_pass DRAM-bound at {INSPIRE_RATE_MULTI_GIBS:.0f} GiB/s with log_p=16 (no padding); "
        f"pack/rgsw multi-core; qGen single-thread):\n"
        f"  dim0={e['dim0']}  t={e['t']}  client_qgen={e['qgen_ms']}ms  "
        f"upload={e['upload_mib']:.2f} MiB→{e['upload_ms']}ms  "
        f"server={e['server_ms']}ms (first_pass={e['first_pass_ms']} + pack={e['pack_ms']} + rgsw={e['rgsw_ms']})  "
        f"download={e['download_kib']:.0f} KiB→{e['download_ms']}ms  decode=0\n"
        f"  total = {e['total_ms']} ms ({e['total_s']:.1f} s)\n"
    )

def format_via_block(e):
    return (
        f"VIA-style total online delay (≥, 70 Mbps; "
        f"FirstDim compute-bound at {VIA_RATE_MULTI_GIBS:.0f} GiB/s on 24c (NTT/ring-poly mul, sublinear scaling from 1.25 GB/s/core); "
        f"+{int(VIA_OVERHEAD_FRAC*100)}% non-FirstDim overhead; "
        f"multi-sub-DB + selector REUSE: {e['native_queries']} sub-DBs × 4 KiB native payload, "
        f"ONE selector applied to all sub-DBs, download per sub-DB):\n"
        f"  LR+LC={e['lr']+e['lc']} ({e['lr']}+{e['lc']})  "
        f"client_qgen={e['qgen_ms']}ms  "
        f"upload={e['upload_mib']:.2f} MiB→{e['upload_ms']}ms  "
        f"server={e['server_ms']}ms (first_pass={e['first_pass_ms']})  "
        f"download={e['download_kib']:.0f} KiB→{e['download_ms']}ms  decode=0\n"
        f"  total = {e['total_ms']} ms ({e['total_s']:.1f} s)\n"
    )

# log-scale (TiB) configurations under NewSummary/
CONFIGS = [
    ('4kb',  range(28, 35), 32768),
    ('8kb',  range(26, 33), 65536),
    ('16kb', range(26, 33), 131072),
    ('64kb', range(24, 31), 524288),
]
OUT_BASE = os.path.join(HERE, "NewSummary")      # default output dir

def item_subdir(item_bits):
    """Map item bytes -> NewSummary subdir name (4kb, 8kb, 16kb, 64kb)."""
    return f"{item_bits // 8 // 1024}kb"

# ──────────────────────────────────────────────────────────────────────────────
# generate: write OO-PIR + extrapolation summary files under NewSummary/
# ──────────────────────────────────────────────────────────────────────────────
def write_summary(path, log2N, item_bits):
    body = (
        format_oopir_block(oopir_estimate(log2N, item_bits)) + '\n'
        + format_simplepir_block(simplepir_estimate(log2N, item_bits)) + '\n'
        + format_inspire_block(inspire_estimate(log2N, item_bits)) + '\n'
        + format_via_block(via_estimate(log2N, item_bits))
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(body)
    print(f"wrote {path}")

def cmd_generate(log2N_range, item_bits, base_dir):
    sub = item_subdir(item_bits)
    for log2N in log2N_range:
        path = os.path.join(base_dir, sub, f"db_log2_{log2N}.log")
        write_summary(path, log2N, item_bits)

# ──────────────────────────────────────────────────────────────────────────────
# bandwidth: total upload+download per (scheme, block, log2N) into NewSummary/bandwidth
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_bytes(b):
    """Render bytes as MiB (3 decimals) -- single unit so columns align."""
    return f"{b / 1024**2:>9.3f} MiB"

ESCAPE_RESULT_BASE = os.path.join(REPO, "result")       # Escape bench logs (main/bench.sh)
RE_ESCAPE_BW = re.compile(
    r'^\s*bandwidth\s*=\s*send\s+([\d.]+)\s*(MiB|KiB).*?recv\s+([\d.]+)\s*(MiB|KiB)',
    re.MULTILINE)

def _escape_bandwidth(block, log2N):
    """Read <repo>/result/<block>/db_log2_<log2N>.log summary line.
       Returns (upload_bytes, download_bytes) or None if not present."""
    p = os.path.join(ESCAPE_RESULT_BASE, block, f'db_log2_{log2N}.log')
    if not os.path.exists(p): return None
    with open(p) as f:
        m = RE_ESCAPE_BW.search(f.read())
    if not m: return None
    def b(v, u): return float(v) * (1024**2 if u == 'MiB' else 1024)
    return b(m.group(1), m.group(2)), b(m.group(3), m.group(4))

def cmd_bandwidth(base_dir):
    """For each scheme, list upload + download = total per (block, log2N)."""
    rows = []  # (scheme, block, log2N, up_b, dn_b)
    for block, log2N_range, item_bits in CONFIGS:
        for log2N in log2N_range:
            oo = oopir_estimate(log2N, item_bits)
            sp = simplepir_estimate(log2N, item_bits)
            ip = inspire_estimate(log2N, item_bits)
            vi = via_estimate(log2N, item_bits)
            esc = _escape_bandwidth(block, log2N)
            rows += [
                ('SimplePIR',  block, log2N, sp['upload_b'],     sp['download_b']),
                ('InsPIRe',    block, log2N, ip['upload_b'],     ip['download_b']),
                ('VIA',        block, log2N, vi['upload_b'],     vi['download_b']),
                ('Piano',      block, log2N, oo['query_bits']/8,    oo['piano_bytes']),
                ('RMS',        block, log2N, oo['query_bits']/8,    oo['rms_bytes']),
            ]
            if esc is not None:
                rows.append(('Escape', block, log2N, esc[0], esc[1]))
    out_lines = ["Bandwidth (upload + download, MiB), per (scheme, block).\n"
                 "log2N reindexed 1..K within each block; each line: (idx, total_MiB).\n"]
    for scheme in ('SimplePIR', 'InsPIRe', 'VIA', 'Piano', 'RMS', 'Escape'):
        out_lines.append(f"=== {scheme} ===")
        # Group rows for this scheme by block, preserving CONFIGS order.
        per_block = {}
        for s, block, log2N, up, dn in rows:
            if s != scheme: continue
            per_block.setdefault(block, []).append(up + dn)
        for block, totals in per_block.items():
            out_lines.append(f"  {block}:")
            for i, t in enumerate(totals):
                out_lines.append(f"    ({i+1}, {t/1024**2:.3f})")
        out_lines.append("")
    path = os.path.join(base_dir, 'bandwidth')
    with open(path, 'w') as f:
        f.write("\n".join(out_lines))
    print(f"wrote {path}")

# ──────────────────────────────────────────────────────────────────────────────
# compute: client_qgen + server time per (scheme, block, log2N) into NewSummary/compute
# ──────────────────────────────────────────────────────────────────────────────
RE_ESCAPE_QCOMP   = re.compile(r'\[query comp\]\s+([\d.]+)\s+ms')
RE_ESCAPE_SANS    = re.compile(r'\[server ans\]\s+([\d.]+)\s+ms')
RE_ESCAPE_SCMP    = re.compile(r'\[server cmp\]\s+([\d.]+)\s+ms')
RE_ESCAPE_ANS_MEM = re.compile(r'mem_stall\s+([\d.]+)\s+ms\s*\(\s*\d+% of answer\)')
RE_ESCAPE_CMP_MEM = re.compile(r'mem_stall\s+([\d.]+)\s+ms\s*\(\s*\d+% of compress\)')

def _escape_compute(block, log2N):
    """Read <repo>/result/<block>/db_log2_<log2N>.log.
       Returns (client_ms, server_ms) where:
         client_ms = [query comp]
         server_ms = ([server ans] - ans mem_stall) + ([server cmp] - cmp mem_stall)
       i.e., the active compute portion only — memory-stall cycles are
       excluded since they reflect DRAM-bandwidth waits, not work."""
    p = os.path.join(ESCAPE_RESULT_BASE, block, f'db_log2_{log2N}.log')
    if not os.path.exists(p): return None
    with open(p) as f: txt = f.read()
    mq  = RE_ESCAPE_QCOMP.search(txt);   ma  = RE_ESCAPE_SANS.search(txt);   mc  = RE_ESCAPE_SCMP.search(txt)
    mam = RE_ESCAPE_ANS_MEM.search(txt); mcm = RE_ESCAPE_CMP_MEM.search(txt)
    if not (mq and ma and mc and mam and mcm): return None
    ans_net = float(ma.group(1)) - float(mam.group(1))
    cmp_net = float(mc.group(1)) - float(mcm.group(1))
    return float(mq.group(1)), ans_net + cmp_net

def cmd_compute(base_dir):
    """For each scheme, list client_qgen_ms + server_ms = total compute per (block, log2N).
       OO-PIR (Piano, RMS) is bandwidth-only (no compute model) → reported as 0."""
    rows = []  # (scheme, block, log2N, client_ms, server_ms)
    for block, log2N_range, item_bits in CONFIGS:
        for log2N in log2N_range:
            sp = simplepir_estimate(log2N, item_bits)
            ip = inspire_estimate(log2N, item_bits)
            vi = via_estimate(log2N, item_bits)
            esc = _escape_compute(block, log2N)
            rows += [
                ('SimplePIR', block, log2N, sp['qgen_ms'], sp['server_ms']),
                ('InsPIRe',   block, log2N, ip['qgen_ms'], ip['server_ms']),
                ('VIA',       block, log2N, vi['qgen_ms'], vi['server_ms']),
                ('Piano',     block, log2N, 0, 0),
                ('RMS',       block, log2N, 0, 0),
            ]
            if esc is not None:
                rows.append(('Escape', block, log2N, esc[0], esc[1]))
    out_lines = ["Compute delay (client_qgen + server, seconds), per (scheme, block).\n"
                 "log2N reindexed 1..K within each block; each line: (idx, total_s).\n"
                 "Piano / RMS = 0 (OO-PIR, bandwidth-only, no compute model).\n"]
    for scheme in ('SimplePIR', 'InsPIRe', 'VIA', 'Piano', 'RMS', 'Escape'):
        out_lines.append(f"=== {scheme} ===")
        per_block = {}
        for s, block, log2N, cli, srv in rows:
            if s != scheme: continue
            per_block.setdefault(block, []).append(cli + srv)
        for block, totals in per_block.items():
            out_lines.append(f"  {block}:")
            for i, t in enumerate(totals):
                out_lines.append(f"    ({i+1}, {t/1000.0:.3f})")
        out_lines.append("")
    path = os.path.join(base_dir, 'compute')
    with open(path, 'w') as f:
        f.write("\n".join(out_lines))
    print(f"wrote {path}")

# ──────────────────────────────────────────────────────────────────────────────
# latency: end-to-end latency per (scheme, block, log2N) into NewSummary/latency
# ──────────────────────────────────────────────────────────────────────────────
RE_TOTAL = {
    'SimplePIR': re.compile(r'^SimplePIR-style.*?\n\s*total = ([\d.]+) ms', re.M | re.S),
    'InsPIRe':   re.compile(r'^InsPIRe-style.*?\n\s*total = ([\d.]+) ms', re.M | re.S),
    'VIA':       re.compile(r'^VIA-style.*?\n\s*total = ([\d.]+) ms', re.M | re.S),
    'Piano':     re.compile(r'total PIANO\s*=.*?=\s*([\d.]+) s'),
    'RMS':       re.compile(r'total RMS\s*=.*?=\s*([\d.]+) s'),
}
RE_ESCAPE_E2E = re.compile(r'^\s*end-to-end\s*=.*=\s*([\d.]+)\s*ms', re.M)

def _escape_latency(block, log2N):
    """Read the 'end-to-end' line of <repo>/result/<block>/db_log2_<log2N>.log (ms):
       query + send + server (no mem_stall) + recv + recover, means over 5 reps."""
    p = os.path.join(ESCAPE_RESULT_BASE, block, f'db_log2_{log2N}.log')
    if not os.path.exists(p): return None
    m = RE_ESCAPE_E2E.search(open(p).read())
    return float(m.group(1)) / 1000 if m else None

def cmd_latency(base_dir):
    """For each scheme, list the end-to-end latency per (block, log2N), taken from the
       per-setting logs written by `generate` (run it first) and from Escape's logs."""
    rows = []  # (scheme, block, log2N, seconds)
    for block, log2N_range, item_bits in CONFIGS:
        for log2N in log2N_range:
            p = os.path.join(base_dir, block, f'db_log2_{log2N}.log')
            assert os.path.exists(p), f"{p} not found (run `generate` first)"
            txt = open(p).read()
            for scheme, rx in RE_TOTAL.items():
                m = rx.search(txt)
                assert m, f"no {scheme} total in {p}"
                val = float(m.group(1))
                rows.append((scheme, block, log2N, val if rx.pattern.endswith(' s') else val / 1000))
            esc = _escape_latency(block, log2N)
            if esc is not None:
                rows.append(('Escape', block, log2N, esc))
    out_lines = ["End-to-end latency (qgen + upload + server + download, seconds), per (scheme, block).\n"
                 "log2N reindexed 1..K within each block; each line: (idx, total_s).\n"
                 f"Piano / RMS: bandwidth-only latency at {NETWORK_MBPS} Mbps (no server compute).\n"
                 "Escape: end-to-end as reported in its logs.\n"]
    for scheme in ('SimplePIR', 'InsPIRe', 'VIA', 'Piano', 'RMS', 'Escape'):
        out_lines.append(f"=== {scheme} ===")
        per_block = {}
        for s, block, log2N, sec in rows:
            if s != scheme: continue
            per_block.setdefault(block, []).append(sec)
        for block, totals in per_block.items():
            out_lines.append(f"  {block}:")
            for i, t in enumerate(totals):
                out_lines.append(f"    ({i+1}, {t:.3f})")
        out_lines.append("")
    path = os.path.join(base_dir, 'latency')
    with open(path, 'w') as f:
        f.write("\n".join(out_lines))
    print(f"wrote {path}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_range(args, default_range):
    if args.log2N is not None:
        return [args.log2N]
    if args.range_str:
        lo, hi = (int(x) for x in args.range_str.split('-'))
        return list(range(lo, hi + 1))
    return list(default_range)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary', default=SUMMARY_LOG,
                   help='measure.py summary file to read the rates from (default: summary.log next to this script)')
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_common(sp, default_item):
        sp.add_argument('--single-thread', action='store_true',
                        help='use the single-thread rates from the summary')
        sp.add_argument('--log2N', type=int, default=None,
                        help='single log2(N) value (e.g. 30)')
        sp.add_argument('--range', dest='range_str', default=None,
                        help='log2N range LO-HI (e.g. 26-32)')
        sp.add_argument('--item-size', type=int, default=default_item,
                        help=f'item size in bytes (default {default_item})')

    sp_g = sub.add_parser('generate', help='write NewSummary/<block>/db_log2_*.log summary files')
    add_common(sp_g, 4096)
    sp_g.add_argument('--base-dir', default=OUT_BASE)

    sp_b = sub.add_parser('bandwidth', help='write NewSummary/bandwidth (upload + download per scheme)')
    sp_b.add_argument('--base-dir', default=OUT_BASE)

    sp_c = sub.add_parser('compute', help='write NewSummary/compute (client_qgen + server per scheme)')
    sp_c.add_argument('--base-dir', default=OUT_BASE)

    sp_l = sub.add_parser('latency', help='write NewSummary/latency (end-to-end latency per scheme; run generate first)')
    sp_l.add_argument('--base-dir', default=OUT_BASE)

    args = p.parse_args()
    load_rates(args.summary)

    if args.cmd == 'generate':
        item_bits = args.item_size * 8
        if args.log2N is None and not args.range_str:
            # No selector: regenerate all log-scale configs (4, 8, 16, 64 KiB).
            for folder, log2N_range, item_bits_cfg in CONFIGS:
                cmd_generate(log2N_range, item_bits_cfg, args.base_dir)
        else:
            log2N_range = _resolve_range(args, [])
            cmd_generate(log2N_range, item_bits, args.base_dir)


    elif args.cmd == 'bandwidth':
        cmd_bandwidth(args.base_dir)

    elif args.cmd == 'compute':
        cmd_compute(args.base_dir)

    elif args.cmd == 'latency':
        cmd_latency(args.base_dir)
