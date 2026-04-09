#!/bin/bash
cd "$(dirname "$0")"

# Exp2: U 系列
for ds in output_n10000_l30_u1 output_n10000_l30_u30 output_n10000_l30_u40 output_n10000_l30_u60 output_n10000_l30_u70 output_n10000_l30_u80 output_n10000_l30_u90 output_n10000_l30_u95 output_n10000_l30_u98 output_n10000_l30_u99; do
  echo ""
  echo "=== $ds === $(date)"
  python run_sequenzo.py -d "$ds"
  python run_sequenzo_other.py -d "$ds"
  python run_tanat.py -d "$ds" -m edit
  python run_tanat_other.py -d "$ds"
  echo "--- cooling 120s ---"
  sleep 120
done

# Exp3: n=5k/20k/30k/50k
for ds in output_n5000_l30_u85 output_n20000_l30_u85 output_n30000_l30_u85 output_n50000_l30_u85; do
  echo ""
  echo "=== $ds === $(date)"
  python run_sequenzo.py -d "$ds"
  python run_sequenzo_other.py -d "$ds"
  python run_tanat.py -d "$ds" -m edit
  python run_tanat_other.py -d "$ds"
  echo "--- cooling 120s ---"
  sleep 120
done

echo ""
echo "ALL DONE $(date)"
