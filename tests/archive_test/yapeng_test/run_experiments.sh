#!/bin/bash
# ============================================================
# run_experiments.sh
#
# Exp1 (S sensitivity, 6 ds):
#   Sequenzo: OM, LCP, LCS, EUCLID
#   TanaT:    OM, LCP, LCS, EUCLID
#
# Exp2 (Robustness, 69 ds):
#   Sequenzo: OM, LCP, LCS, EUCLID, OMspell, OMtspell, HAM, DHD
#
# 75 unique datasets | Est. ~6h on M3 Air (plugged in)
#
# Results (results/):
#   results_sqz_om.csv      — Sequenzo OM
#   results_sqz_other.csv   — Sequenzo LCP/LCS/EUCLID
#   results_sqz_extra.csv   — Sequenzo OMspell/OMtspell/HAM/DHD (Exp2)
#   results_tat.csv         — TanaT (Exp1)
#
# Usage:
#   ./run_experiments.sh           # run all
#   ./run_experiments.sh gen       # generate datasets only
#   ./run_experiments.sh resume    # skip completed datasets
# ============================================================

set -e

# ---- Paths (edit if needed) ----
GEN_DIR=~/Desktop/Sequenzo/tests/archive_test/xinyi_test/expe_tanat/random-sequences-generation
WORK_DIR=~/Desktop/Sequenzo/tests/archive_test/yapeng_test
OUT_DIR="$WORK_DIR/generated_datasets"
RES_DIR="$WORK_DIR/results"

ITERS=3
MODE=${1:-run}

mkdir -p "$RES_DIR"

# ============================================================
# PART A: Dataset enumeration (75 unique)
# ============================================================
ALL_DATASETS=()
add() { ALL_DATASETS+=("$1,$2,$3,$4"); }

# Exp1: S sensitivity
for s in 5 10 15 20 25 30; do add 10000 30 85 $s; done

# Exp2a: Fix S={5,20,30}
for s in 5 20 30; do
    for l in 10 30 50 100;    do add 10000 $l 85 $s; done
    for u in 20 50 85 100;    do add 10000 30 $u $s; done
    for n in 1000 3000 10000; do add $n    30 85 $s; done
done

# Exp2b: Fix n={1k,10k,30k}
for n in 1000 10000 30000; do
    for l in 10 30 50 100;     do add $n $l 85 20; done
    for u in 20 50 85 100;     do add $n 30 $u 20; done
    for s in 5 10 15 20 25 30; do add $n 30 85 $s; done
done

# Exp2c: Fix L={10,30,100}
for l in 10 30 100; do
    for n in 1000 3000 10000;  do add $n    $l 85 20; done
    for u in 20 50 85 100;     do add 10000 $l $u 20; done
    for s in 5 10 15 20 25 30; do add 10000 $l 85 $s; done
done

# Exp2d: Fix U={20,85,100}
for u in 20 85 100; do
    for n in 1000 3000 10000;  do add $n    30 $u 20; done
    for l in 10 30 50 100;     do add 10000 $l $u 20; done
    for s in 5 10 15 20 25 30; do add 10000 30 $u $s; done
done

UNIQUE=($(printf '%s\n' "${ALL_DATASETS[@]}" | sort -t, -k1,1n -k2,2n -k3,3n -k4,4n | uniq))
echo "Unique datasets: ${#UNIQUE[@]}"

# Exp1 check: n=10000, L=30, U=85, any S
is_exp1_entry() {
    IFS=',' read -r _n _l _u _s <<< "$1"
    [[ "$_n" == "10000" && "$_l" == "30" && "$_u" == "85" ]]
}

# ============================================================
# PART B: Dataset generation
# ============================================================

# Existing S=20 datasets (old naming without _s suffix)
LEGACY_COMBOS="1000,30,85 2000,30,85 3000,30,85 4000,30,85 5000,30,85 \
10000,10,85 10000,30,1 10000,30,20 10000,30,30 \
10000,30,40 10000,30,50 10000,30,60 10000,30,70 \
10000,30,80 10000,30,85 10000,30,90 10000,30,95 \
10000,30,98 10000,30,99 10000,30,100 \
10000,50,85 10000,100,85 10000,200,85 10000,500,85 \
20000,30,85 30000,30,85 50000,30,85"

generate_all() {
    echo ""
    echo "========================================"
    echo "  GENERATING DATASETS"
    echo "========================================"

    # Symlink existing S=20 datasets to new naming
    for combo in $LEGACY_COMBOS; do
        IFS=',' read -r n l u <<< "$combo"
        local new="output_n${n}_l${l}_u${u}_s20"
        local old="output_n${n}_l${l}_u${u}"
        if [ -d "$OUT_DIR/$old" ] && [ ! -e "$OUT_DIR/$new" ]; then
            ln -s "$old" "$OUT_DIR/$new"
            echo "[LINK] $new -> $old"
        fi
    done

    local gen=0 skip=0
    for entry in "${UNIQUE[@]}"; do
        IFS=',' read -r n l u s <<< "$entry"
        local name="output_n${n}_l${l}_u${u}_s${s}"
        if [ -e "$OUT_DIR/$name" ]; then
            skip=$((skip + 1)); continue
        fi
        echo "[GEN] $name"
        mkdir -p "$OUT_DIR/$name"
        (cd "$GEN_DIR" && python generation.py -t random \
            -n "$n" -l "$l" -d "$s" --U="$u" --pd \
            -o "$OUT_DIR/$name/${name}.dat")
        gen=$((gen + 1))
    done
    echo "[GEN] Done. New: $gen | Exist: $skip"
}

# ============================================================
# PART C: Benchmark
# ============================================================

SQZ_OM="$RES_DIR/results_sqz_om.csv"
SQZ_OTHER="$RES_DIR/results_sqz_other.csv"
SQZ_EXTRA="$RES_DIR/results_sqz_extra.csv"
TAT_CSV="$RES_DIR/results_tat.csv"

SQZ_OM_H="it,method,nb,voc,mean_len,max_dur,metric,cl,prep_time,time,ARI"
PARSED_H="it,method,dataset,nb,time_points,metric,time"
TAT_H="it,method,nb,voc,mean_len,max_dur,metric,cl,prep_time,time,ARI"

init_csvs() {
    if [[ "$MODE" == "resume" ]]; then
        [ ! -f "$SQZ_OM" ]    && echo "$SQZ_OM_H" > "$SQZ_OM"
        [ ! -f "$SQZ_OTHER" ] && echo "$PARSED_H"  > "$SQZ_OTHER"
        [ ! -f "$SQZ_EXTRA" ] && echo "$PARSED_H"  > "$SQZ_EXTRA"
        [ ! -f "$TAT_CSV" ]   && echo "$TAT_H"    > "$TAT_CSV"
    else
        echo "$SQZ_OM_H" > "$SQZ_OM"
        echo "$PARSED_H"  > "$SQZ_OTHER"
        echo "$PARSED_H"  > "$SQZ_EXTRA"
        echo "$TAT_H"    > "$TAT_CSV"
    fi
}

get_ds() {
    local n=$1 l=$2 u=$3 s=$4
    local name="output_n${n}_l${l}_u${u}_s${s}"
    [ -e "$OUT_DIR/$name" ] && echo "$name" && return
    if [ "$s" -eq 20 ]; then
        local old="output_n${n}_l${l}_u${u}"
        [ -e "$OUT_DIR/$old" ] && echo "$old" && return
    fi
    echo ""
}

is_done() {
    [[ "$MODE" != "resume" ]] && return 1
    grep -q "$1" "$SQZ_OM" 2>/dev/null
}

# ---- Parse stdout → CSV, while also printing timing lines to terminal ----
parse_and_show() {
    local output="$1"
    local iter="$2"
    local csv_file="$3"

    local ds_name n_seq tp
    ds_name=$(echo "$output" | grep "^Dataset:" | head -1 | sed 's/Dataset: \([^,]*\).*/\1/')
    n_seq=$(echo "$output"   | grep "^Dataset:" | head -1 | sed 's/.*n=\([0-9]*\).*/\1/')
    tp=$(echo "$output"      | grep "^Dataset:" | head -1 | sed 's/.*time_points=\([0-9]*\).*/\1/')

    # Print timing lines to terminal + write to CSV
    echo "$output" | grep "\[Sequenzo\].*time_elapsed" | while IFS= read -r line; do
        echo "    $line"   # ← show in terminal
        local metric elapsed
        metric=$(echo "$line" | awk '{print $2}')
        elapsed=$(echo "$line" | awk '{print $NF}')
        echo "$iter,Sequenzo,$ds_name,$n_seq,$tp,$metric,$elapsed" >> "$csv_file"
    done

    # Print errors if any
    echo "$output" | grep -i "ERROR" | while IFS= read -r line; do
        echo "    [!] $line"
    done
}

# ---- Run functions ----

run_sqz_om() {
    local ds=$1 iter=$2
    # stdout → terminal (includes [Sequenzo] time_elapsed)
    # stderr → show (catch errors)
    python run_sequenzo.py -d "$ds" -i "$iter" || true
    tail -n +2 results_sqz.csv >> "$SQZ_OM" 2>/dev/null || true
}

run_sqz_other() {
    local ds=$1 iter=$2
    local output
    output=$(python run_sequenzo_other.py -d "$ds" 2>&1) || true
    parse_and_show "$output" "$iter" "$SQZ_OTHER"
}

run_sqz_extra() {
    local ds=$1 iter=$2
    local output
    output=$(python run_sequenzo_extra.py -d "$ds" 2>&1) || true
    parse_and_show "$output" "$iter" "$SQZ_EXTRA"
}

run_tat() {
    local ds=$1 iter=$2 metric=$3
    # stdout → terminal (includes [TanaT] time_elapsed)
    python run_tanat.py -d "$ds" -m "$metric" -i "$iter" || true
    tail -n +2 results.csv >> "$TAT_CSV" 2>/dev/null || true
}

# ============================================================
# PART D: Main loop
# ============================================================

run_all() {
    init_csvs

    local total=${#UNIQUE[@]}
    local idx=0
    local GLOBAL_START=$(date +%s)

    echo ""
    echo "========================================"
    echo "  BENCHMARKS"
    echo "  Machine: $(uname -m), $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null) cores"
    echo "  Datasets: $total | Iters: $ITERS"
    echo "  Exp1 (6 ds):  Sqz(OM LCP LCS EUCLID) + TanaT(OM LCP LCS EUCLID)"
    echo "  Exp2 (69 ds): Sqz(OM LCP LCS EUCLID OMspell OMtspell HAM DHD)"
    echo "  Results: $RES_DIR/"
    echo "========================================"
    echo ""

    for entry in "${UNIQUE[@]}"; do
        IFS=',' read -r n l u s <<< "$entry"
        idx=$((idx + 1))

        local ds=$(get_ds $n $l $u $s)
        if [ -z "$ds" ]; then
            echo "[$idx/$total] [MISS] n=$n L=$l U=$u S=$s — dataset not found"
            continue
        fi
        if is_done "$ds"; then
            echo "[$idx/$total] [SKIP] $ds"
            continue
        fi

        local is_exp1=""
        is_exp1_entry "$entry" && is_exp1="yes"

        local DS_START=$(date +%s)

        echo "──────────────────────────────────────────"
        if [ -n "$is_exp1" ]; then
            echo "[$idx/$total] $ds  ▸ Exp1: Sqz + TanaT"
        else
            echo "[$idx/$total] $ds  ▸ Exp2: Sqz (8 metrics)"
        fi
        echo "──────────────────────────────────────────"

        cd "$WORK_DIR"

        for i in $(seq 1 $ITERS); do
            echo "  ── iter $i/$ITERS ──"

            # Sequenzo OM (stdout shows timing directly)
            run_sqz_om "$ds" "$i"

            # Sequenzo LCP / LCS / EUCLID (parsed + shown)
            run_sqz_other "$ds" "$i"

            if [ -n "$is_exp1" ]; then
                # TanaT: OM, LCP, LCS, EUCLID (stdout shows timing directly)
                run_tat "$ds" "$i" "edit"
                run_tat "$ds" "$i" "lcp"
                run_tat "$ds" "$i" "lcs"
                run_tat "$ds" "$i" "linear"
            else
                # Sequenzo extra: OMspell, OMtspell, HAM, DHD (parsed + shown)
                run_sqz_extra "$ds" "$i"
            fi
        done

        local DS_END=$(date +%s)
        local DS_SEC=$((DS_END - DS_START))
        echo "  ✓ $ds done in ${DS_SEC}s"
        echo ""

        # Progress every 10 datasets
        if (( idx % 10 == 0 )); then
            local NOW=$(date +%s)
            local ELAPSED_MIN=$(( (NOW - GLOBAL_START) / 60 ))
            local ETA_MIN=$(( (total - idx) * (NOW - GLOBAL_START) / idx / 60 ))
            echo "═══ Progress: $idx/$total | ${ELAPSED_MIN}min elapsed | ~${ETA_MIN}min remaining ═══"
            echo ""
        fi
    done

    local GLOBAL_END=$(date +%s)
    local TOTAL_SEC=$((GLOBAL_END - GLOBAL_START))
    local TOTAL_MIN=$((TOTAL_SEC / 60))
    local TOTAL_REMAIN=$((TOTAL_SEC % 60))

    echo "════════════════════════════════════════"
    echo "  ALL DONE — ${TOTAL_MIN}min ${TOTAL_REMAIN}s"
    echo "════════════════════════════════════════"
    echo ""
    echo "Result files:"
    wc -l "$SQZ_OM" "$SQZ_OTHER" "$SQZ_EXTRA" "$TAT_CSV" 2>/dev/null
    echo ""
    echo "CSV columns:"
    echo "  sqz_om:    $SQZ_OM_H"
    echo "  sqz_other: $PARSED_H"
    echo "  sqz_extra: $PARSED_H"
    echo "  tat:       $TAT_H"
}

# ============================================================
# MAIN
# ============================================================
generate_all

if [[ "$MODE" == "gen" ]]; then
    echo "Generation complete."
    exit 0
fi

run_all
