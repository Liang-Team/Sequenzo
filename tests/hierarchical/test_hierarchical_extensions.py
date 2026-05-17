"""
Tests for hierarchical extensions: additive/crossed ANOVA, clustering, residuals, sampling.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sequenzo.hierarchical import (
    make_relational_sequences,
    compute_relational_distance_matrix,
    crossed_sequence_discrepancy,
    hierarchical_sequence_discrepancy,
    check_interaction_identifiability,
    compute_pair_residuals,
    detect_pair_specific_outliers,
    cluster_pair_sequences,
    cluster_pair_trajectories,
    sample_pairwise_distances,
    summarize_distance_by_structure_sampled,
    describe_sampling_scheme,
    run_hierarchical_sequence_analysis,
    encode_states,
    validate_relational_sequence_data,
)


def _toy_long_data(n_regions=2, n_cpc=2):
    rows = []
    for ri in range(n_regions):
        region = f"R{ri + 1}"
        for ci in range(n_cpc):
            cpc = f"C{ci + 1}"
            for t, state in enumerate([0, 0, 1, 2]):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state if region == "R1" else 0,
                    }
                )
    return pd.DataFrame(rows)


def _dist(n_regions=2, n_cpc=2):
    df = _toy_long_data(n_regions, n_cpc)
    seq = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    return seq, compute_relational_distance_matrix(seq, method="HAM")


def test_additive_is_default_decomposition():
    _, dist = _dist()
    decomp = hierarchical_sequence_discrepancy(dist)
    assert decomp.additive is not None
    assert decomp.crossed is None
    a = decomp.additive
    assert abs(a.joint_share + a.residual_share - 1.0) < 1e-6
    assert a.joint_share >= 0


def test_summary_documents_non_exclusive_marginals():
    df = _toy_long_data()
    result = run_hierarchical_sequence_analysis(
        df,
        level_1_col="region_id",
        level_2_col="cpc_id",
        time_col="year",
        state_col="state",
        distance_method="HAM",
        n_perm=0,
    )
    text = result.summary()
    assert "Marginal pseudo-R²" in text or "marginal" in text.lower()
    assert "Joint explained" in text or "joint" in text.lower()
    assert "non-exclusive" in text.lower() or "Additive decomposition" in text


def test_crossed_warns_when_saturated():
    _, dist = _dist()
    info = check_interaction_identifiability(
        dist.level_1_ids, dist.level_2_ids
    )
    assert info["saturated"] is True
    with pytest.warns(UserWarning, match="saturated"):
        crossed_sequence_discrepancy(dist)


def test_pair_residuals_additive_default():
    seq, dist = _dist()
    res = compute_pair_residuals(seq, dist)
    assert "standardized_residual" in res.columns
    outliers = detect_pair_specific_outliers(seq, dist, top_n=4)
    assert "is_outlier" in outliers.columns


def test_sampling_index_alignment_with_subsample():
    """i_index must index into full-length level_1_ids after sequence subsampling."""
    df = _toy_long_data(n_regions=4, n_cpc=4)
    seq = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    original_n = seq.n_pairs
    sample = sample_pairwise_distances(
        seq,
        n_pair_samples=50,
        max_sequences=8,
        sampling_unit="sequence",
        random_state=0,
    )
    assert sample.n_pairs == original_n
    assert len(sample.level_1_ids) == original_n
    assert len(sample.level_2_ids) == original_n
    assert sample.i_index.max() < original_n
    assert sample.j_index.max() < original_n
  # must not raise
    _ = sample.level_1_ids[sample.i_index]
    _ = sample.level_2_ids[sample.j_index]
    struct = summarize_distance_by_structure_sampled(sample)
    assert len(struct) == 4


def test_sampling_pair_mode():
    seq, _ = _dist()
    sample = sample_pairwise_distances(
        seq,
        n_pair_samples=20,
        sampling_unit="pair",
        method="HAM",
        random_state=1,
    )
    assert sample.sampling_unit == "pair"
    assert sample.n_pairs == 4
    assert len(sample.level_1_ids) == 4
    assert sample.n_sampled <= 20
    text = describe_sampling_scheme(sample)
    assert "direct pair sampling" in text


def test_describe_sampling_sequence_mode():
    seq, _ = _dist()
    sample = sample_pairwise_distances(
        seq, n_pair_samples=5, max_sequences=3, sampling_unit="sequence"
    )
    text = sample.describe()
    assert "sequence subsampling" in text


def test_clustering_and_pipeline():
    seq, dist = _dist()
    pair_cl = cluster_pair_sequences(dist, k=2, verbose=False)
    assert len(pair_cl.cluster_labels) == 4

    typology = cluster_pair_trajectories(
        seq,
        k=2,
        algorithm="pam",
        distance_matrix=dist,
        verbose=False,
    )
    assert typology.level == "pair"
    assert len(typology.cluster_labels) == 4
    assert typology.distance_to_medoids.shape == (4, 2)

    clara_typology = cluster_pair_trajectories(
        seq,
        k=2,
        algorithm="clara",
        n_iterations=5,
        sample_size=4,
        verbose=False,
        aggregate_identical=False,
        random_state=0,
    )
    assert clara_typology.method == "CLARA"
    assert len(clara_typology.cluster_labels) == 4

    df = _toy_long_data()
    result = run_hierarchical_sequence_analysis(
        df,
        level_1_col="region_id",
        level_2_col="cpc_id",
        time_col="year",
        state_col="state",
        distance_method="HAM",
        n_perm=0,
        cluster_k=2,
    )
    assert result.decomposition.additive is not None
    text = result.summary()
    assert "Additive decomposition" in text


def test_encode_states_empty():
    codes, alphabet = encode_states([])
    assert codes.shape == (0, 0)
    assert alphabet == []


def test_mismatched_time_grid_raises():
    rows = [
        {"region_id": "R1", "cpc_id": "C1", "year": 2000, "state": 0},
        {"region_id": "R1", "cpc_id": "C1", "year": 2001, "state": 0},
        {"region_id": "R1", "cpc_id": "C2", "year": 2001, "state": 0},
        {"region_id": "R1", "cpc_id": "C2", "year": 2002, "state": 1},
    ]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="time grid"):
        validate_relational_sequence_data(
            df, "region_id", "cpc_id", "year", "state"
        )


def test_plot_additive_component_shares_deprecated(close_mpl_figures):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from sequenzo.hierarchical import plot_additive_component_shares

    _, dist = _dist()
    decomp = hierarchical_sequence_discrepancy(dist)
    with pytest.warns(DeprecationWarning):
        ax = plot_additive_component_shares(decomp)
    if ax is not None:
        plt.close(ax.figure)
