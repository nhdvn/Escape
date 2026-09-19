#!/bin/bash
# Run bench skip-gen for all (block, log2_db) configs and append a summary
# block to each log. Output base dir defaults to result/ but can be overridden:
#   OUT_BASE=result_rerun main/bench.sh   # write to result_rerun/ instead
cd "$(dirname "$0")/.."

OUT_BASE="${OUT_BASE:-result}"

CONFIGS="
4kb 28 1024 512 512
4kb 29 512 1024 1024
4kb 30 1024 1024 1024
4kb 31 2048 1024 1024
4kb 32 1024 2048 2048
4kb 33 2048 2048 2048
4kb 34 1024 4096 4096
8kb 26 256 512 512
8kb 27 512 512 512
8kb 28 1024 512 512
8kb 29 512 1024 1024
8kb 30 1024 1024 1024
8kb 31 2048 1024 1024
8kb 32 1024 2048 2048
16kb 26 256 512 512
16kb 27 512 512 512
16kb 28 1024 512 512
16kb 29 512 1024 1024
16kb 30 1024 1024 1024
16kb 31 2048 1024 1024
16kb 32 1024 2048 2048
64kb 24 256 256 256
64kb 25 512 256 256
64kb 26 256 512 512
64kb 27 512 512 512
64kb 28 1024 512 512
64kb 29 512 1024 1024
64kb 30 1024 1024 1024
"

mapfile -t LINES < <(echo "$CONFIGS" | grep .)
total=${#LINES[@]}

for i in "${!LINES[@]}"; do
    read block log2 n1 mr1 mc1 <<< "${LINES[$i]}"
    bk=${block%kb}
    mkdir -p "$OUT_BASE/$block"
    out=$OUT_BASE/$block/db_log2_$log2.log
    echo "[$((i+1))/$total] $block log2=$log2  N1=$n1 MR1=$mr1 MC1=$mc1  -> $out"

    if ! make EXTRA="-DBENCH_SKIP_GEN=1 -DBLOCK_KIB=$bk -DPLHE_N1=$n1 -DPLHE_MR1=$mr1 -DPLHE_MC1=$mc1" -B > /tmp/build_$$.log 2>&1; then
        echo "BUILD FAILED at $block log2=$log2"
        cat /tmp/build_$$.log | tail -5
        continue
    fi
    ./bench 5 > "$out" 2>&1 || true
    # Append (idempotently) the means summary block to this log.
    python3 -c "import sys; sys.path.insert(0,'main'); import summary; summary.update_log('$out')"
    grep -E "srv delay|^OK$|^FAIL" "$out" | tail -3
done
rm -f /tmp/build_$$.log
echo DONE
