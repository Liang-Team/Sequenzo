#!/bin/bash
# ============================================================
# Ablation Experiment: Post-optimization
# ============================================================
#
# Design
# ------
# 2x2 factorial: (Pre/Post optimization) x (OpenMP On/Off)
# This script: Post-optimization only (Groups C & D)
#   Group C: OMP_NUM_THREADS=1  (optimized single-thread C++)
#   Group D: default            (optimized C++ + OpenMP parallel)
#
# Three sub-experiments, each varying one factor:
#   Ablation-n: n = 1k, 5k, 10k       (L=30, U=85)
#   Ablation-U: U = 30, 50, 85         (n=10k, L=30)
#   Ablation-L: L = 10, 30, 50         (n=10k, U=85)
#
# Shared baseline: output_n10000_l30_u85 (appears in all three)
#
# 7 unique datasets, 42 runs total
# Machine: MacBook Air M3 24GB (fanless, 120s cooldown)
# ============================================================

cd "$(dirname "$0")"

COOLDOWN=120

# ---- Helper functions ----

run_omp_off() {
    local ds=$1 id=$2
    echo ""
    echo "===== [$ds] C${id}: OMP OFF ===== $(date)"
    OMP_NUM_THREADS=1 python run_sequenzo_omvariants.py -d "$ds"
    sleep $COOLDOWN
}

run_omp_on() {
    local ds=$1 id=$2
    echo ""
    echo "===== [$ds] D${id}: OMP ON ===== $(date)"
    python run_sequenzo_omvariants.py -d "$ds"
    sleep $COOLDOWN
}

full_cd() {
    # 3x C + 3x D for a dataset
    local ds=$1
    for i in 1 2 3; do run_omp_off "$ds" $i; done
    for i in 1 2 3; do run_omp_on  "$ds" $i; done
}

# ---- Main ----

echo "============================================================"
echo "Ablation Post-optimization"
echo "Start: $(date)"
echo "============================================================"

# Ablation-n: Sample Size (L=30, U=85)
echo ""
echo "############## Ablation-n: Sample Size ##############"

echo ""; echo "------ n=1,000 ------"
full_cd output_n1000_l30_u85

echo ""; echo "------ n=5,000 ------"
full_cd output_n5000_l30_u85

echo ""; echo "------ n=10,000 ------"
full_cd output_n10000_l30_u85

# Ablation-U: Uniqueness Rate (n=10k, L=30)
# U=85 shared with Ablation-n above
echo ""
echo "############## Ablation-U: Uniqueness Rate ##############"

echo ""; echo "------ U=30 ------"
full_cd output_n10000_l30_u30

echo ""; echo "------ U=50 ------"
full_cd output_n10000_l30_u50

# Ablation-L: Sequence Length (n=10k, U=85)
# L=30 shared with Ablation-n above
echo ""
echo "############## Ablation-L: Sequence Length ##############"

echo ""; echo "------ L=10 ------"
full_cd output_n10000_l10_u85

echo ""; echo "------ L=50 ------"
full_cd output_n10000_l50_u85

echo ""
echo "============================================================"
echo "Post-optimization ablation complete: $(date)"
echo ""
echo "  Ablation-n:  n=1k (C3D3)  n=5k (C3D3)  n=10k (C3D3)"
echo "  Ablation-U:  U=30 (C3D3)  U=50 (C3D3)  U=85  (shared)"
echo "  Ablation-L:  L=10 (C3D3)  L=30 (shared) L=50  (C3D3)"
echo ""
echo "Next: compare A/B (pre) vs C/D (post) for ablation analysis"
echo "============================================================"
