#!/bin/bash
# Run bench skip-gen for the 12 partition shapes in result_/: N = 2^30 entries of
# 8, 16 and 64 KiB, split as N1 x MR1 x MC1 with N1 = 1024 ... 65536, and append
# a summary block to each log. Output dir defaults to result_/ but can be overridden:
#   OUT_BASE=result_rerun offline/bench.sh
cd "$(dirname "$0")/.."

OUT_BASE="${OUT_BASE:-result_}"

# block  N1     MR1   MC1        (N1 * MR1 * MC1 = 2^30)
CONFIGS="
8kb   1024  1024  1024
8kb   4096  512   512
8kb   16384 256   256
8kb   65536 128   128
16kb  1024  1024  1024
16kb  4096  512   512
16kb  16384 256   256
16kb  65536 128   128
64kb  1024  1024  1024
64kb  4096  512   512
64kb  16384 256   256
64kb  65536 128   128
"

mapfile -t LINES < <(echo "$CONFIGS" | grep .)
total=${#LINES[@]}
mkdir -p "$OUT_BASE"

for i in "${!LINES[@]}"; do
    read block n1 mr1 mc1 <<< "${LINES[$i]}"
    bk=${block%kb}
    out=$OUT_BASE/${block}_30_n${n1}.log
    echo "[$((i+1))/$total] $block  N1=$n1 MR1=$mr1 MC1=$mc1  -> $out"

    if ! make EXTRA="-DBENCH_SKIP_GEN=1 -DBLOCK_KIB=$bk -DPLHE_N1=$n1 -DPLHE_MR1=$mr1 -DPLHE_MC1=$mc1" -B > /tmp/build_$$.log 2>&1; then
        echo "BUILD FAILED at $block N1=$n1"
        tail -5 /tmp/build_$$.log
        continue
    fi
    ./bench 5 > "$out" 2>&1 || true
    # Append (idempotently) the means summary block to this log.
    python3 -c "import sys; sys.path.insert(0,'main'); import summary; summary.update_log('$out')"
    grep -E "srv delay|^OK$|^FAIL" "$out" | tail -3
done
rm -f /tmp/build_$$.log
echo DONE
