#!/bin/bash
# ============================================================
# Ablation Experiment: Pre-optimization Baseline
# ============================================================
#
# Design
# ------
# 2x2 factorial: (Pre/Post optimization) x (OpenMP On/Off)
# This script: Pre-optimization only (Groups A & B)
#   Group A: OMP_NUM_THREADS=1  (pure single-thread C++)
#   Group B: default            (C++ + OpenMP parallel)
#
# Three sub-experiments, each varying one factor:
#   Ablation-n: n = 1k, 5k, 10k       (L=30, U=85)
#   Ablation-U: U = 30, 50, 85         (n=10k, L=30)
#   Ablation-L: L = 10, 30, 50         (n=10k, U=85)
#
# Shared baseline: output_n10000_l30_u85 (appears in all three)
#   - A1/A2 already completed in previous session
#   - This script runs A3 + B1/B2/B3
#
# 7 unique datasets, 40 runs total, ~10 hours
# Machine: MacBook Air M3 24GB (fanless, 120s cooldown)
# ============================================================

set -e
cd "$(dirname "$0")"

COOLDOWN=120

# ---- Helper functions ----

run_omp_off() {
    local ds=$1 id=$2
    echo ""
    echo "===== [$ds] A${id}: OMP OFF ===== $(date)"
    OMP_NUM_THREADS=1 python run_sequenzo_omvariants.py -d "$ds"
    sleep $COOLDOWN
}

run_omp_on() {
    local ds=$1 id=$2
    echo ""
    echo "===== [$ds] B${id}: OMP ON ===== $(date)"
    python run_sequenzo_omvariants.py -d "$ds"
    sleep $COOLDOWN
}

full_ab() {
    # 3x A + 3x B for a dataset
    local ds=$1
    for i in 1 2 3; do run_omp_off "$ds" $i; done
    for i in 1 2 3; do run_omp_on  "$ds" $i; done
}

# ---- Main ----

echo "============================================================"
echo "Ablation Pre-optimization Baseline"
echo "Start: $(date)"
echo "============================================================"

# Ablation-n: Sample Size (L=30, U=85)
echo ""
echo "############## Ablation-n: Sample Size ##############"

echo ""; echo "------ n=1,000 ------"
full_ab output_n1000_l30_u85

echo ""; echo "------ n=5,000 ------"
full_ab output_n5000_l30_u85

echo ""; echo "------ n=10,000 (A3 + B1/B2/B3) ------"
run_omp_off output_n10000_l30_u85 3
for i in 1 2 3; do run_omp_on output_n10000_l30_u85 $i; done

# Ablation-U: Uniqueness Rate (n=10k, L=30)
# U=85 shared with Ablation-n above
echo ""
echo "############## Ablation-U: Uniqueness Rate ##############"

echo ""; echo "------ U=30 ------"
full_ab output_n10000_l30_u30

echo ""; echo "------ U=50 ------"
full_ab output_n10000_l30_u50

# Ablation-L: Sequence Length (n=10k, U=85)
# L=30 shared with Ablation-n above
echo ""
echo "############## Ablation-L: Sequence Length ##############"

echo ""; echo "------ L=10 ------"
full_ab output_n10000_l10_u85

echo ""; echo "------ L=50 ------"
full_ab output_n10000_l50_u85

echo ""
echo "============================================================"
echo "Pre-optimization baseline complete: $(date)"
echo ""
echo "  Ablation-n:  n=1k (A3B3)  n=5k (A3B3)  n=10k (A1A2+A3 B3)"
echo "  Ablation-U:  U=30 (A3B3)  U=50 (A3B3)  U=85  (shared)"
echo "  Ablation-L:  L=10 (A3B3)  L=30 (shared) L=50  (A3B3)"
echo ""
echo "Next: optimize C++ code, recompile, run ablation_post.sh"
echo "============================================================"
