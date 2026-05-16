#!/usr/bin/env python3
"""
Validating sequence-analysis typologies for subsequent regression.

Python walkthrough mirroring Matthias Studer's WeightedCluster short R tutorial
(``clustassoc`` vignette on the mvad data). Steps and outputs are aligned with:

  library(TraMineR); library(fastcluster); library(WeightedCluster)
  set.seed(1); data(mvad); seqdef(...); seqdist(LCS); hclust(ward.D)
  as.clustrange(...); clustassoc(...); plot(clustassoc, main = "Unaccounted")

Run from the repository root::

    python Tutorials/feature_extraction_and_selection/clustassoc_typology_validation_tutorial.py

Optional: regenerate R reference tables (requires R with TraMineR and WeightedCluster)::

    Rscript tests/clustering/weightedcluster_reference_clustassoc_mvad.R

The pytest module ``tests/clustering/test_clustassoc_mvad_r_parity.py`` checks
numerical agreement with those references.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the script directly from this folder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sequenzo import SequenceData, get_distance_matrix, load_dataset, plot_sequence_index
from sequenzo.clustering import (
    cluster_association,
    hierarchical_cluster_range,
    plot_cluster_association,
)

# Reproducibility (R tutorial uses set.seed(1); clustering here is deterministic).
np.random.seed(1)

# -----------------------------------------------------------------------------
# Step 1 — Load mvad and define the state sequence object (TraMineR: seqdef)
# -----------------------------------------------------------------------------
print("\n[Step 1] Load mvad and build SequenceData (TraMineR seqdef)...")

df = load_dataset("mvad")

# R: seqdef(mvad, 17:86, ...) — columns 17–86 in R = index 16:86 in Python (70 months).
TIME_COLUMNS = list(df.columns[16:86])

ALPHABET = ["employment", "FE", "HE", "joblessness", "school", "training"]
STATE_LABELS = [
    "employment",
    "further education",
    "higher education",
    "joblessness",
    "school",
    "training",
]

seqdata = SequenceData(
    df,
    time=TIME_COLUMNS,
    id_col="id",
    states=ALPHABET,
    labels=STATE_LABELS,
)

# -----------------------------------------------------------------------------
# Step 2 — Pairwise dissimilarities (TraMineR: seqdist, method = "LCS")
# -----------------------------------------------------------------------------
print("\n[Step 2] LCS distance matrix (TraMineR seqdist)...")

diss = np.asarray(get_distance_matrix(seqdata, method="LCS"), dtype=float)
print(f"  Distance matrix shape: {diss.shape}")

# -----------------------------------------------------------------------------
# Step 3 — Hierarchical clustering (R: hclust(..., method = "ward.D"))
#     and cluster-quality range (WeightedCluster: as.clustrange)
# -----------------------------------------------------------------------------
print("\n[Step 3] Ward D hierarchical clustering and CQI table (as.clustrange)...")

clustrange = hierarchical_cluster_range(diss, maxcluster=10, method="ward.d")
clustqual = clustrange.stats
print(clustqual.round(4))

# -----------------------------------------------------------------------------
# Step 4 — Association between typology and covariate (WeightedCluster: clustassoc)
# -----------------------------------------------------------------------------
print("\n[Step 4] clustassoc: father unemployment (funemp) vs trajectories...")

covariate = df["funemp"].to_numpy()
cla = cluster_association(clustrange, diss, covariate)
print(cla.round(7))

# -----------------------------------------------------------------------------
# Step 5 — Plot unaccounted association (WeightedCluster: plot(clustassoc))
# -----------------------------------------------------------------------------
print("\n[Step 5] Plot Unaccounted share (plot.clustassoc)...")

plot_cluster_association(cla, stat="Unaccounted", title="Unaccounted", show=False)
plt.savefig(
    _REPO_ROOT
    / "Tutorials"
    / "feature_extraction_and_selection"
    / "clustassoc_unaccounted.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("  Saved: Tutorials/feature_extraction_and_selection/clustassoc_unaccounted.png")

# -----------------------------------------------------------------------------
# Step 6 — Compare k = 5 and k = 6 index plots (TraMineR: seqdplot)
# -----------------------------------------------------------------------------
print("\n[Step 6] Index plots for 5 vs 6 clusters (seqdplot)...")

for num_clusters in (5, 6):
    column = f"cluster{num_clusters}"
    membership = clustrange.clustering[[column]].copy()
    membership.columns = ["Cluster"]
    membership["id"] = df["id"].to_numpy()

    out_path = (
        _REPO_ROOT
        / "Tutorials"
        / "feature_extraction_and_selection"
        / f"clustassoc_index_plot_k{num_clusters}.png"
    )
    plot_sequence_index(
        seqdata,
        group_dataframe=membership,
        group_column_name="Cluster",
        title=f"Index plot — {num_clusters} clusters",
        save_as=str(out_path),
        dpi=150,
    )
    plt.close("all")
    print(f"  Saved: {out_path}")

print("\n[Done] Tutorial finished. Compare printed tables with the R vignette output.")
