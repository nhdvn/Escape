#!/usr/bin/env python3
"""Estimate per-query bandwidth for Escape vs Piano and RMS.

Writes two tables to bandwidth.log (next to this script), one row per entry
(block) size of 2^x bytes, x in DEFAULT_EXPS:

    Table 1: fixed database size DB_TIB TiB, so N = DB_bytes / block entries
    Table 2: fixed number of entries N = 2^FIXED_LOG_N

and two hint-update tables (2^FIXED_LOG_N entries of 8 KiB and of 64 KiB), one row
per number of online queries Q = 2^q, q in QUERY_EXPS:

    Table 3 / 4: average bandwidth per query = max(DB / Q, per-query cost)

where DB / Q is the full-database download of a hint refresh amortized over the
Q queries it serves, and the per-query cost is the same as in Tables 1 and 2.

Escape parameters use a cube split (N1 partitions x MR1 rows x MC1 cols):
MC1 = MR1 = smallest power of two >= cbrt(N), and N1 = N / (MC1*MR1) <= MC1.
Only N1 and MC1 enter the bandwidth; MR1 is shown for completeness.

Piano and RMS are sqrt-hint schemes (offline/online), so their costs are driven by
sqrt(N). Formulas taken from ExpSummary/64kb/db_log2_30.log:

    Piano download = sqrt(N) * block
    RMS   download = 2 * sqrt(N) / log2(sqrt(N)) * block
    index (both)   = sqrt(N) * log2(sqrt(N))  bits         (upload)

Escape send/recv byte formulas are from client/client.c (send_bytes /
recv_bytes) and compress/_m512.c (n_slots / n_out packing):

    ct        = pl_bits * 2 / 8               bytes per Paillier ciphertext (mod N^2)
    send      = LAMBDA*ct + N1*4 + N1*MC1*4    (zkey + sub_rows + enc_bits)
    recv      = n_out*ct
    headroom  = ceil_log2(LAMBDA + 1)
    log_r     = LOG_P + headroom + 2
    n_slots   = (pl_bits - headroom) // log_r
    n_out     = ceil(B_CHUNKS / n_slots)
    B_CHUNKS  = block rounded up to an even number of P_BITS(=8)-bit chunks
"""
import math
import os

# ---- fixed system parameters -------------------------------------------------
DB_TIB   = 64          # logical database size (TiB)
LAMBDA   = 1024        # LWE secret dimension (PLHE_LAMBDA)
PL_BITS  = 2048        # Paillier modulus bits (cntx->pl_bits)
LOG_P    = 8           # plaintext modulus bits (PLHE_P_BITS)

DB_BYTES = DB_TIB * (1 << 40)
CT_BYTES = PL_BITS * 2 // 8            # one Paillier ciphertext on the wire
MiB      = 1024.0 * 1024.0
DEFAULT_EXPS = (3, 6, 9, 11, 12, 14, 16, 18)   # block sizes 2^x bytes in both tables
FIXED_LOG_N  = 30                              # Tables 2-4: N = 2^30 entries
QUERY_EXPS   = (11, 15, 19, 24, 30, 35)       # Tables 3-4: Q = 2^q online queries


def ceil_log2(x):
    return (x - 1).bit_length()        # matches ceil_log2_u32 for x >= 1


def escape_split(N):
    """Cube split with MC1 = MR1 and N1 <= MC1.

    MC1 = MR1 = smallest power of two >= cbrt(N) (so mc^3 >= N), and
    N1 = N / (MC1*MR1) is whatever is left -- guaranteed N1 <= MC1.
    e.g. N = 2^43 -> MC1 = MR1 = 2^15, N1 = 2^13 (N1*MC1*MR1 = N)."""
    mc = 1
    while mc * mc * mc < N:
        mc <<= 1
    MC1 = MR1 = mc
    N1  = max(1, N // (mc * mc))
    return N1, MC1, MR1


def b_chunks(block_bytes):
    """B_CHUNKS for an entry of block_bytes, replicating plhe/params.h."""
    block_bits = block_bytes * 8
    n_p = (block_bits + LOG_P - 1) // LOG_P     # P_BITS-bit chunks needed
    return ((n_p + 1) // 2) * 2                 # round up to even (SIMD)


def escape_bandwidth(N, block_bytes):
    """Return (total_MiB, N1, MC1, MR1, send_B, recv_B) for Escape."""
    N1, MC1, MR1 = escape_split(N)

    send = LAMBDA * CT_BYTES + N1 * 4 + N1 * MC1 * 4

    bc       = b_chunks(block_bytes)
    headroom = ceil_log2(LAMBDA + 1)
    log_r    = LOG_P + headroom + 2
    n_slots  = (PL_BITS - headroom) // log_r
    n_out    = (bc + n_slots - 1) // n_slots
    recv     = n_out * CT_BYTES

    return (send + recv) / MiB, N1, MC1, MR1, send, recv


def piano_bandwidth(N, block_bytes):
    """Return (piano_MiB, rms_MiB) for Piano and RMS."""
    sqn = round(math.isqrt(N) if N > 0 else 0)
    lg  = max(1, ceil_log2(sqn))                # log2(sqrt(N)) bits

    index_bytes = sqn * lg / 8                  # upload, shared by both
    dl1 = sqn * block_bytes                     # Piano download
    dl2 = (2.0 * sqn / lg) * block_bytes        # RMS download

    total1 = (dl1 + index_bytes) / MiB
    total2 = (dl2 + index_bytes) / MiB
    return total1, total2


def table(title, n_for):
    """One table: a row per block size 2^x (x in DEFAULT_EXPS); n_for(block) gives N."""
    hdr = (f"{'2^x':>5} {'bytes':>8} {'N':>16} {'N1':>8} {'MC1':>8} {'MR1':>8} "
           f"{'Escape':>10} {'Piano':>12} {'RMS':>12}")
    lines = [title, "", hdr, "-" * len(hdr)]
    for b in DEFAULT_EXPS:
        block = 1 << b
        N = n_for(block)
        esc, N1, MC1, MR1, send, recv = escape_bandwidth(N, block)
        p1, p2 = piano_bandwidth(N, block)
        lines.append(f"{'2^' + str(b):>5} {block:>8} {N:>16} {N1:>8} {MC1:>8} {MR1:>8} "
                     f"{esc:>10.3f} {p1:>12.3f} {p2:>12.3f}")
    return lines


def hint_table(title, block):
    """Average bandwidth per query when a hint refresh (full-DB download) is
       amortized over Q = 2^q online queries: max(DB / Q, per-query cost)."""
    N = 1 << FIXED_LOG_N
    db_mib = N * block / MiB
    esc = escape_bandwidth(N, block)[0]
    piano, rms = piano_bandwidth(N, block)
    hdr = (f"{'Q':>5} {'DB/Q':>12} {'Escape':>10} {'Piano':>12} {'RMS':>12}")
    lines = [title, f"per-query cost: Escape {esc:.3f}, Piano {piano:.3f}, RMS {rms:.3f}", "",
             hdr, "-" * len(hdr)]
    for q in QUERY_EXPS:
        off = db_mib / (1 << q)
        lines.append(f"{'2^' + str(q):>5} {off:>12.3f} {max(off, esc):>10.3f} "
                     f"{max(off, piano):>12.3f} {max(off, rms):>12.3f}")
    return lines


def main():
    lines = [f"Per-query bandwidth (MiB), LAMBDA={LAMBDA}, pl_bits={PL_BITS}", ""]
    lines += table(f"Table 1: fixed DB = {DB_TIB} TiB (N = DB / bytes)", lambda block: DB_BYTES // block)
    lines += [""]
    lines += table(f"Table 2: fixed N = 2^{FIXED_LOG_N} entries", lambda block: 1 << FIXED_LOG_N)
    lines += [""]
    lines += hint_table(f"Table 3: hint-update bandwidth per query (MiB), (2^{FIXED_LOG_N} x 8 KiB) DB", 8 * 1024)
    lines += [""]
    lines += hint_table(f"Table 4: hint-update bandwidth per query (MiB), (2^{FIXED_LOG_N} x 64 KiB) DB", 64 * 1024)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bandwidth.log")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
