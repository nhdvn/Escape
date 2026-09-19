#!/usr/bin/env python3
"""Estimate per-query bandwidth for Escape vs Piano (1) and Piano (2).

Sweeps entry (block) size from 2^3 .. 2^16 bytes (powers of two). For each
block size the logical database is fixed at DB_TIB TiB, so the entry count is

    N = DB_bytes / block

Escape parameters use a cube-root split (the scheme is 3-dimensional:
N1 partitions x MR1 rows x MC1 cols), so

    N1 = MC1 = round(N ** (1/3)),   MR1 = N / (N1 * MC1)

Only N1 and MC1 enter the bandwidth; MR1 is shown for completeness.

Piano is a sqrt-hint scheme (offline/online), so its costs are driven by
sqrt(N). Formulas taken from ExpSummary/64kb/db_log2_30.log:

    Piano (1) download = sqrt(N) * block
    Piano (2) download = 2 * sqrt(N) / log2(sqrt(N)) * block   (RMS variant)
    Piano index (both) = sqrt(N) * log2(sqrt(N))  bits         (upload)

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
import argparse
import math

# ---- fixed system parameters -------------------------------------------------
DB_TIB   = 64          # logical database size (TiB)
LAMBDA   = 1024        # LWE secret dimension (PLHE_LAMBDA)
PL_BITS  = 2048        # Paillier modulus bits (cntx->pl_bits)
LOG_P    = 8           # plaintext modulus bits (PLHE_P_BITS)

DB_BYTES = DB_TIB * (1 << 40)
CT_BYTES = PL_BITS * 2 // 8            # one Paillier ciphertext on the wire
MiB      = 1024.0 * 1024.0


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
    """Return (total1_MiB, total2_MiB) for Piano (1) and Piano (2)."""
    sqn = round(math.isqrt(N) if N > 0 else 0)
    lg  = max(1, ceil_log2(sqn))                # log2(sqrt(N)) bits

    index_bytes = sqn * lg / 8                  # upload, shared by both
    dl1 = sqn * block_bytes                     # Piano (1) download
    dl2 = (2.0 * sqn / lg) * block_bytes        # Piano (2) / RMS download

    total1 = (dl1 + index_bytes) / MiB
    total2 = (dl2 + index_bytes) / MiB
    return total1, total2


def main():
    ap = argparse.ArgumentParser(description="Per-query bandwidth estimate.")
    ap.add_argument("--fixed-n", type=int, metavar="LOG2", default=None,
                    help="fix N = 2^LOG2 for every block (default: N = 64 TiB / block)")
    ap.add_argument("--min-b", type=int, default=3, help="smallest block = 2^min-b bytes")
    ap.add_argument("--max-b", type=int, default=16, help="largest block = 2^max-b bytes")
    args = ap.parse_args()

    print(f"Per-query bandwidth (MiB)")
    if args.fixed_n is not None:
        print(f"Fixed N = 2^{args.fixed_n}, LAMBDA={LAMBDA}, pl_bits={PL_BITS}\n")
    else:
        print(f"Fixed DB = {DB_TIB} TiB, LAMBDA={LAMBDA}, pl_bits={PL_BITS}\n")
    hdr = (f"{'block':>8} {'N':>16} {'N1':>8} {'MC1':>8} {'MR1':>8} "
           f"{'Escape':>10} {'Piano(1)':>12} {'Piano(2)':>12}")
    print(hdr)
    print("-" * len(hdr))
    for b in range(args.min_b, args.max_b + 1):
        block = 1 << b
        N = (1 << args.fixed_n) if args.fixed_n is not None else DB_BYTES // block
        esc, N1, MC1, MR1, send, recv = escape_bandwidth(N, block)
        p1, p2 = piano_bandwidth(N, block)
        print(f"{block:>8} {N:>16} {N1:>8} {MC1:>8} {MR1:>8} "
              f"{esc:>10.3f} {p1:>12.3f} {p2:>12.3f}")


if __name__ == "__main__":
    main()
