#!/usr/bin/env python3
"""
Studer (2018) fuzzy + property-based clustering workflow (biofam).

Mirrors the WeightedCluster short R tutorial:
  seqdef -> seqdist(LCS) -> fanny -> summary(membership)
  -> seqpropclust(state, duration) -> as.clustrange

Run from repo root:
    python3 -u Tutorials/cluster_analysis/test_studer_2018_workflow.py
    python3 -u Tutorials/cluster_analysis/test_studer_2018_workflow.py --full

Default uses n=400 sequences (fast sanity check). --full uses all 2000 rows;
FANNY alone may take 30-60+ minutes in pure Python (R cluster::fanny ~35 s).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sequenzo import SequenceData, get_distance_matrix, load_dataset
from sequenzo.clustering import (
    get_fuzzy_clusters,
    membership_summary,
    print_property_tree,
    property_based_clustering,
    property_clustering_quality,
)

# R tutorial reference targets (WeightedCluster on biofam, set.seed(1))
R_MEMBERSHIP_SUMMARY_MEAN = {
    "V1": 0.205773,
    "V2": 0.211530,
    "V3": 0.210022,
    "V4": 0.178295,
    "V5": 0.194380,
}
R_PROPERTY_GLOBAL_R2 = 0.48761
R_CLUSTER_QUALITY = pd.DataFrame(
    {
        "PBC": [0.50, 0.54, 0.54, 0.58],
        "HG": [0.61, 0.69, 0.73, 0.80],
        "R2": [0.21, 0.34, 0.42, 0.49],
    },
    index=["cluster2", "cluster3", "cluster4", "cluster5"],
)

STATE_LABELS = [
    "Parent",
    "Left",
    "Married",
    "Left/Married",
    "Child",
    "Left/Child",
    "Left/Married/Child",
    "Divorced",
]
TIME_COLS = [str(age) for age in range(15, 31)]


def load_biofam_sequences(n_rows: int | None) -> SequenceData:
    df = load_dataset("biofam").reset_index(drop=True)
    if n_rows is not None:
        df = df.head(n_rows)
    df["id"] = np.arange(len(df))
    return SequenceData(
        df,
        time=TIME_COLS,
        states=list(range(8)),
        labels=STATE_LABELS,
        id_col="id",
    )


def _compare_table(label: str, got: pd.DataFrame, ref: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n=== {label} (Sequenzo vs R reference) ===")
    for col in cols:
        if col not in got.columns or col not in ref.columns:
            continue
        g = got[col].to_numpy(dtype=float)
        r = ref[col].to_numpy(dtype=float)
        diff = np.abs(g - r)
        print(f"  {col}: max |diff| = {diff.max():.4f}, mean |diff| = {diff.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Studer 2018 biofam workflow test")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all 2000 biofam sequences (slow FANNY)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=400,
        help="Subsample size when not using --full (default: 400)",
    )
    args = parser.parse_args()
    n_rows = None if args.full else args.n

    print(f"[>] Loading biofam ({'full' if args.full else f'n={n_rows}'})...")
    t0 = time.perf_counter()
    seqdata = load_biofam_sequences(n_rows)
    print(f"    {seqdata.seqdata.shape[0]} sequences, {len(TIME_COLS)} time points")

    print("[>] LCS distance matrix...")
    t1 = time.perf_counter()
    diss = get_distance_matrix(seqdata, method="LCS")
    diss = np.asarray(diss.values if hasattr(diss, "values") else diss, dtype=float)
    print(f"    done in {time.perf_counter() - t1:.1f}s; mean diss = {diss.mean():.4f}")

    print("[>] FANNY fuzzy clustering (k=5, memb.exp=1.5)...")
    t2 = time.perf_counter()
    fclust = get_fuzzy_clusters(diss, n_clusters=5, memb_exp=1.5, method="fanny")
    print(
        f"    done in {time.perf_counter() - t2:.1f}s; "
        f"converged={fclust.converged}, iterations={fclust.iterations}"
    )

    summary = membership_summary(fclust.membership)
    print("\n=== membership summary (Mean row) ===")
    print(summary.loc[["Mean"]].round(6))
    if args.full:
        mean_row = summary.loc["Mean"]
        for col in R_MEMBERSHIP_SUMMARY_MEAN:
            if col in mean_row.index:
                ref = R_MEMBERSHIP_SUMMARY_MEAN[col]
                got = float(mean_row[col])
                print(f"  {col}: got={got:.6f}, R={ref:.6f}, diff={abs(got - ref):.6f}")

    print("\n[>] Property-based clustering (state + duration, max_clusters=5)...")
    t3 = time.perf_counter()
    pclust = property_based_clustering(
        seqdata,
        diss=diss,
        properties=["state", "duration"],
        max_clusters=5,
        verbose=True,
    )
    print(f"    done in {time.perf_counter() - t3:.1f}s")
    print_property_tree(pclust)

    adj = pclust["info"].get("adjustment") or {}
    r2 = adj.get("R2")
    if r2 is not None:
        print(f"\nGlobal R2: {r2:.5f}", end="")
        if args.full:
            print(f"  (R reference: {R_PROPERTY_GLOBAL_R2:.5f})")
        else:
            print()

    print("\n[>] Cluster quality (as.clustrange)...")
    pclustqual = property_clustering_quality(pclust, diss=diss, n_clusters=5)
    print(pclustqual.stats.round(2))
    if args.full:
        _compare_table("cluster quality", pclustqual.stats, R_CLUSTER_QUALITY, ["PBC", "HG", "R2"])

    print(f"\n[>] Total elapsed: {time.perf_counter() - t0:.1f}s")
    if not args.full:
        print(
            "[i] Subsample run — use --full for R parity checks on membership / R2 / quality."
        )


if __name__ == "__main__":
    main()
