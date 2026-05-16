"""
Parity tests: Studer clustassoc vignette on mvad (TraMineR + WeightedCluster reference).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix, load_dataset
from sequenzo.clustering import (
    cluster_association,
    hierarchical_cluster_range,
    plot_cluster_association,
)

REF_DIR = Path(__file__).resolve().parent / "reference_data"
RTOL = 1e-5
ATOL = 1e-4


def _mvad_sequence_data() -> SequenceData:
    """Match TraMineR ``seqdef(mvad, 17:86, ...)`` in the clustassoc vignette."""
    df = load_dataset("mvad")
    time_cols = list(df.columns[16:86])
    alphabet = ["employment", "FE", "HE", "joblessness", "school", "training"]
    labels = [
        "employment",
        "further education",
        "higher education",
        "joblessness",
        "school",
        "training",
    ]
    return SequenceData(
        df,
        time=time_cols,
        id_col="id",
        states=alphabet,
        labels=labels,
    )


@pytest.fixture(scope="module")
def r_reference():
    if not (REF_DIR / "ref_mvad_clustassoc.csv").exists():
        pytest.skip(
            "R reference files missing. Run: "
            "Rscript tests/clustering/weightedcluster_reference_clustassoc_mvad.R"
        )
    return {
        "diss": pd.read_csv(REF_DIR / "ref_mvad_lcs_diss.csv", index_col=0).values,
        "clustassoc": pd.read_csv(REF_DIR / "ref_mvad_clustassoc.csv", index_col=0),
        "clustrange_stats": pd.read_csv(REF_DIR / "ref_mvad_clustrange_stats.csv", index_col=0),
        "clustering": pd.read_csv(REF_DIR / "ref_mvad_clustrange_clustering.csv", index_col=0),
    }


def test_mvad_lcs_distance_matches_r(r_reference):
    seqdata = _mvad_sequence_data()
    diss_py = np.asarray(get_distance_matrix(seqdata, method="LCS"), dtype=float)
    np.testing.assert_allclose(diss_py, r_reference["diss"], rtol=0, atol=0)


def test_hierarchical_cluster_range_matches_r_partitions(r_reference):
    clustrange = hierarchical_cluster_range(r_reference["diss"], maxcluster=10, method="ward.d")
    for column in r_reference["clustering"].columns:
        py = clustrange.clustering[column].to_numpy()
        ref = r_reference["clustering"][column].to_numpy()
        assert np.array_equal(py, ref), f"Partition mismatch for {column}"


def test_clustassoc_table_matches_r(r_reference):
    clustrange = hierarchical_cluster_range(r_reference["diss"], maxcluster=10, method="ward.d")
    df = load_dataset("mvad")
    cla = cluster_association(clustrange, r_reference["diss"], df["funemp"].to_numpy())
    ref = r_reference["clustassoc"]
    for col in ["Unaccounted", "Remaining", "BIC"]:
        np.testing.assert_allclose(
            cla[col].to_numpy(dtype=float),
            ref[col].to_numpy(dtype=float),
            rtol=RTOL,
            atol=ATOL,
            err_msg=col,
        )


def test_clustrange_cqi_matches_r(r_reference):
    clustrange = hierarchical_cluster_range(r_reference["diss"], maxcluster=10, method="ward.d")
    ref = r_reference["clustrange_stats"]
    for column in ["PBC", "HG", "R2"]:
        np.testing.assert_allclose(
            clustrange.stats[column].to_numpy(dtype=float),
            ref[column].to_numpy(dtype=float),
            rtol=RTOL,
            atol=ATOL,
            err_msg=column,
        )


def test_plot_cluster_association_runs():
    table = pd.DataFrame(
        {
            "Unaccounted": [1.0, 0.68, 0.45],
            "Remaining": [0.006, 0.004, 0.003],
            "BIC": [642.0, 639.0, 642.0],
            "numcluster": [1, 2, 3],
        },
        index=["No Clustering", "cluster2", "cluster3"],
    )
    ax = plot_cluster_association(table, show=False)
    assert ax is not None
