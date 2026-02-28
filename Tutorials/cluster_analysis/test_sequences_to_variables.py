#!/usr/bin/env python3
"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqs2vars_utils.py
@Time    : 01/03/2026 01:30
@Desc    :
Test script for sequences-to-variables (Helske et al. 2024) in cluster_analysis context.

Uses the mvad dataset and the same pipeline as mvad_cluster_analysis.ipynb to ensure
representativeness, hard/soft classification variables, and pseudoclass work end-to-end.

Run from repo root or from this directory:
    python Tutorials/cluster_analysis/test_sequences_to_variables.py
    python test_sequences_to_variables.py  # when cwd is Tutorials/cluster_analysis
"""
import sys
import os
import numpy as np

# Ensure repo root is on path when running from Tutorials/cluster_analysis
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sequenzo import load_dataset, SequenceData, get_distance_matrix, KMedoids

from sequenzo.clustering.seqs2vars_utils import max_distance, cluster_labels_to_dummies
from sequenzo.clustering.sequences_to_variables import (
    representativeness_matrix,
    medoid_indices_from_kmedoids_result,
    cluster_labels_from_kmedoids_result,
    hard_classification_variables,
    fanny_membership,
    soft_classification_variables,
    pseudoclass_regression,
)


def get_mvad_data_and_diss():
    """Load mvad and compute distance matrix (same as mvad_cluster_analysis.ipynb)."""
    df = load_dataset("mvad")
    time_list = [
        "Jul.93", "Aug.93", "Sep.93", "Oct.93", "Nov.93", "Dec.93",
        "Jan.94", "Feb.94", "Mar.94", "Apr.94", "May.94", "Jun.94", "Jul.94",
        "Aug.94", "Sep.94", "Oct.94", "Nov.94", "Dec.94", "Jan.95", "Feb.95",
        "Mar.95", "Apr.95", "May.95", "Jun.95", "Jul.95", "Aug.95", "Sep.95",
        "Oct.95", "Nov.95", "Dec.95", "Jan.96", "Feb.96", "Mar.96", "Apr.96",
        "May.96", "Jun.96", "Jul.96", "Aug.96", "Sep.96", "Oct.96", "Nov.96",
        "Dec.96", "Jan.97", "Feb.97", "Mar.97", "Apr.97", "May.97", "Jun.97",
        "Jul.97", "Aug.97", "Sep.97", "Oct.97", "Nov.97", "Dec.97", "Jan.98",
        "Feb.98", "Mar.98", "Apr.98", "May.98", "Jun.98", "Jul.98", "Aug.98",
        "Sep.98", "Oct.98", "Nov.98", "Dec.98", "Jan.99", "Feb.99", "Mar.99",
        "Apr.99", "May.99", "Jun.99",
    ]
    states = ["FE", "HE", "employment", "joblessness", "school", "training"]
    state_labels = [
        "further education", "higher education", "employment",
        "joblessness", "school", "training",
    ]
    sequence_data = SequenceData(
        df,
        time=time_list,
        id_col="id",
        states=states,
        labels=state_labels,
    )
    om = get_distance_matrix(sequence_data, method="OM", sm="CONSTANT", indel=1)
    # Ensure numpy for KMedoids
    if hasattr(om, "values"):
        diss = np.asarray(om.values, dtype=float)
    else:
        diss = np.asarray(om, dtype=float)
    return sequence_data, diss


def test_phase0_and_phase1(sequence_data, diss):
    """Phase 0: max_distance, cluster_labels_to_dummies. Phase 1: representativeness_matrix."""
    n = diss.shape[0]
    d_max = max_distance(diss)
    assert d_max > 0 and np.isfinite(d_max), "max_distance should be positive and finite"
    print(f"  [OK] max_distance(diss) = {d_max:.4f}")

    labels = np.array([0, 1, 0, 2, 1], dtype=int)
    dummies = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert dummies.shape == (5, 2), "dummies shape (n, K-1)"
    print("  [OK] cluster_labels_to_dummies(labels, k=3, reference=0)")

    # Use a small k for speed; take first 100 rows to keep test fast
    n_use = min(100, n)
    diss_small = diss[:n_use, :n_use]
    k = 4
    kmed = KMedoids(diss_small, k=k, method="PAMonce", verbose=False)
    medoids = medoid_indices_from_kmedoids_result(kmed)
    assert len(medoids) == k, "K medoid indices"
    assert medoids.min() >= 0 and medoids.max() < n_use
    print(f"  [OK] KMedoids + medoid_indices_from_kmedoids_result -> {k} medoids")

    R = representativeness_matrix(diss_small, medoids, d_max=None)
    assert R.shape == (n_use, k)
    assert np.all(R >= 0) and np.all(R <= 1)
    # Each medoid has R=1 for its own column
    for j, med in enumerate(medoids):
        assert np.isclose(R[med, j], 1.0), f"medoid {med} column {j} should have R=1"
    print(f"  [OK] representativeness_matrix(diss, medoids) shape {R.shape}, R in [0,1]")

    R_df = representativeness_matrix(
        diss_small, medoids, d_max=None,
        ids=sequence_data.ids[:n_use], as_dataframe=True,
    )
    assert list(R_df.columns) == [f"R_{j+1}" for j in range(k)]
    assert len(R_df) == n_use
    print("  [OK] representativeness_matrix(..., as_dataframe=True, ids=...)")
    return diss_small, kmed, medoids, n_use, k


def test_phase2(diss_small, kmed, k):
    """Phase 2: hard_classification_variables and cluster_labels_from_kmedoids_result."""
    cluster_labels = cluster_labels_from_kmedoids_result(kmed)
    assert cluster_labels.min() >= 0 and cluster_labels.max() < k
    assert len(np.unique(cluster_labels)) == k

    dummies = hard_classification_variables(cluster_labels, k=k, reference=0)
    n_use = diss_small.shape[0]
    assert dummies.shape == (n_use, k - 1)
    assert np.all((dummies == 0) | (dummies == 1))
    print("  [OK] cluster_labels_from_kmedoids_result + hard_classification_variables")
    return cluster_labels


def test_phase3(diss_small, k):
    """Phase 3: fanny_membership and soft_classification_variables."""
    U, medoids = fanny_membership(diss_small, k=k, m=1.4, random_state=42)
    n_use = diss_small.shape[0]
    assert U.shape == (n_use, k)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(U >= 0) and np.all(U <= 1)

    X_soft = soft_classification_variables(U, reference=0)
    assert X_soft.shape == (n_use, k - 1)
    print("  [OK] fanny_membership + soft_classification_variables")
    return U


def test_phase4(U, n_use):
    """Phase 4: pseudoclass_regression (requires statsmodels)."""
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [SKIP] pseudoclass_regression (statsmodels not installed)")
        return
    np.random.seed(123)
    y = np.random.randn(n_use) * 2 + 1
    out = pseudoclass_regression(y, U, X_fixed=None, M=5, reference=0, random_state=42)
    assert "beta_combined" in out and "se_combined" in out
    # Design matrix has K-1 dummy columns (reference omitted)
    assert len(out["beta_combined"]) == U.shape[1] - 1
    assert len(out["se_combined"]) == U.shape[1] - 1
    print("  [OK] pseudoclass_regression(y, U, M=5)")
    return


def main():
    print("=" * 60)
    print("Sequences-to-variables (Helske et al. 2024) test")
    print("Tutorials/cluster_analysis")
    print("=" * 60)

    print("\n1. Loading mvad and computing distance matrix...")
    sequence_data, diss = get_mvad_data_and_diss()
    n = diss.shape[0]
    print(f"   Loaded {n} sequences, diss shape {diss.shape}")

    print("\n2. Phase 0 & 1: max_distance, dummies, representativeness_matrix...")
    diss_small, kmed, medoids, n_use, k = test_phase0_and_phase1(sequence_data, diss)

    print("\n3. Phase 2: hard classification variables...")
    test_phase2(diss_small, kmed, k)

    print("\n4. Phase 3: FANNY membership and soft classification variables...")
    U = test_phase3(diss_small, k)

    print("\n5. Phase 4: pseudoclass_regression...")
    test_phase4(U, n_use)

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
