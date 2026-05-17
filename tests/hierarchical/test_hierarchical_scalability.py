"""Scalable modes, CLARA wrapper, compression, and structural sampling."""

import numpy as np
import pytest

from sequenzo.hierarchical import (
    compress_identical_relational_sequences,
    cluster_pair_typology_clara,
    make_relational_sequences,
    run_hierarchical_sequence_analysis,
    sample_pairwise_distances,
    sample_structural_pairwise_distances,
    hierarchical_sequence_discrepancy_from_sample,
)


def _toy_long_data():
    rows = []
    for ri, region in enumerate(["R1", "R2"]):
        for ci, cpc in enumerate(["C1", "C2"]):
            seq = [0, 0, 1, 2] if region == "R1" else [0, 0, 0, 0]
            for t, state in enumerate(seq):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state,
                    }
                )
    import pandas as pd

    return pd.DataFrame(rows)


def test_compress_identical_relational_sequences():
    seq = make_relational_sequences(
        _toy_long_data(), "region_id", "cpc_id", "year", "state", validate=False
    )
    compressed = compress_identical_relational_sequences(seq)
    assert compressed.original_n_pairs == 4
    assert compressed.compressed_data.n_pairs == 2
    assert compressed.weights.sum() == pytest.approx(4.0)
    assert compressed.compressed_data.records[0].pair_id.startswith("pattern_")
    assert compressed.details["compressed_pattern_ids"] is True
    assert len(compressed.pattern_id_to_representative_pair) == 2


def test_sample_pairwise_distances_structural_unit():
    seq = make_relational_sequences(
        _toy_long_data(), "region_id", "cpc_id", "year", "state", validate=False
    )
    sample = sample_pairwise_distances(
        seq,
        sampling_unit="structural",
        n_same_level_1=5,
        n_same_level_2=5,
        n_baseline=10,
        random_state=0,
    )
    assert sample.sampling_unit == "structural"
    assert sample.n_sampled > 0


def test_typology_only_plot_distance_heatmap_raises():
    import pandas as pd

    rows = []
    for ri, region in enumerate(["R1", "R2", "R3"]):
        for ci, cpc in enumerate(["C1", "C2", "C3"]):
            seq = [(ri + ci) % 3] * 4
            for t, state in enumerate(seq):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state,
                    }
                )
    result = run_hierarchical_sequence_analysis(
        pd.DataFrame(rows),
        "region_id",
        "cpc_id",
        "year",
        "state",
        analysis_mode="typology_only",
        cluster_k=2,
        typology_algorithm="clara",
        typology_n_iterations=5,
        typology_sample_size=12,
        random_state=0,
        n_perm=0,
        run_outliers=False,
    )
    with pytest.raises(ValueError, match="full distance matrix"):
        result.plot_distance_heatmap()


def test_typology_only_skips_full_matrix():
    import pandas as pd

    rows = []
    for ri, region in enumerate(["R1", "R2", "R3"]):
        for ci, cpc in enumerate(["C1", "C2", "C3"]):
            seq = [(ri + ci) % 3] * 4
            for t, state in enumerate(seq):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state,
                    }
                )
    df = pd.DataFrame(rows)
    result = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "cpc_id",
        "year",
        "state",
        analysis_mode="typology_only",
        cluster_k=2,
        typology_algorithm="clara",
        typology_n_iterations=10,
        typology_sample_size=12,
        random_state=0,
        n_perm=0,
        run_outliers=False,
    )
    assert result.distance_matrix is None
    assert result.pair_typology is not None
    assert result.decomposition is None
    text = result.summary()
    assert "Pair-level typology" in text
    assert "Full distance matrix stored: False" in text


def _level1_planted_long_data():
    """Regions share one trajectory; technologies differ only across regions."""
    import pandas as pd

    region_seq = {
        "R1": [1, 1, 2, 2],
        "R2": [0, 0, 1, 1],
        "R3": [2, 2, 0, 0],
    }
    rows = []
    for region, seq in region_seq.items():
        for tech in ["T1", "T2", "T3"]:
            for t, state in enumerate(seq):
                rows.append(
                    {
                        "region_id": region,
                        "tech_id": tech,
                        "year": 2000 + t,
                        "state": state,
                    }
                )
    return pd.DataFrame(rows)


def test_analysis_modes_exact_sampled_typology_smoke():
    """Three pipeline modes: exact matrix, sampled decomposition, typology-only CLARA."""
    import pandas as pd

    df = _level1_planted_long_data()
    exact = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "tech_id",
        "year",
        "state",
        analysis_mode="exact",
        n_perm=0,
        run_outliers=False,
        random_state=0,
    )
    assert exact.distance_matrix is not None
    assert exact.decomposition is not None

    sampled = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "tech_id",
        "year",
        "state",
        analysis_mode="sampled",
        n_same_level_1=100,
        n_same_level_2=100,
        n_baseline_pairs=200,
        n_perm=0,
        run_outliers=False,
        random_state=0,
    )
    assert sampled.distance_matrix is None
    assert sampled.sampled_distances is not None
    assert sampled.decomposition is not None

    typo = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "tech_id",
        "year",
        "state",
        analysis_mode="typology_only",
        cluster_k=2,
        typology_algorithm="clara",
        typology_n_iterations=8,
        typology_sample_size=15,
        n_perm=0,
        run_outliers=False,
        random_state=0,
    )
    assert typo.distance_matrix is None
    assert typo.pair_typology is not None

    with pytest.raises(ValueError, match="Decomposition plot"):
        typo.plot_decomposition()


def test_exact_vs_sampled_marginal_direction_consistency():
    """Exact and structural-sample modes should agree on which level dominates."""
    df = _level1_planted_long_data()
    exact = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "tech_id",
        "year",
        "state",
        analysis_mode="exact",
        n_perm=0,
        run_outliers=False,
        random_state=0,
    )
    sampled = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "tech_id",
        "year",
        "state",
        analysis_mode="sampled",
        n_same_level_1=200,
        n_same_level_2=200,
        n_baseline_pairs=400,
        n_perm=0,
        run_outliers=False,
        random_state=0,
    )
    assert exact.distance_matrix is not None
    assert exact.decomposition is not None
    assert exact.decomposition.method != "structural_sample"
    assert sampled.distance_matrix is None
    assert sampled.decomposition is not None
    assert sampled.decomposition.method == "structural_sample"

    e1, e2 = exact.decomposition.level_1.pseudo_r2, exact.decomposition.level_2.pseudo_r2
    s1, s2 = sampled.decomposition.level_1.pseudo_r2, sampled.decomposition.level_2.pseudo_r2
    assert e1 > e2, "planted level-1 structure should dominate in exact mode"
    assert s1 > s2, "sampled contrasts should preserve level-1 dominance direction"
    assert (e1 - e2) * (s1 - s2) > 0, "exact and sampled agree on dominant level sign"


def test_sampled_mode_structural_decomposition():
    seq = make_relational_sequences(
        _toy_long_data(), "region_id", "cpc_id", "year", "state", validate=False
    )
    sample = sample_structural_pairwise_distances(
        seq,
        n_same_level_1=10,
        n_same_level_2=10,
        n_baseline=20,
        random_state=0,
    )
    assert sample.sampling_unit == "structural"
    assert sample.n_sampled > 0
    decomp = hierarchical_sequence_discrepancy_from_sample(sample)
    assert decomp.method == "structural_sample"
    assert len(decomp.structural_summary) >= 3


def test_clara_medoid_indices_zero_based():
    import pandas as pd

    rows = []
    for ri, region in enumerate(["R1", "R2", "R3"]):
        for ci, cpc in enumerate(["C1", "C2", "C3"]):
            base = (ri + ci) % 3
            seq = [base] * 4
            for t, state in enumerate(seq):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state,
                    }
                )
    seq = make_relational_sequences(
        pd.DataFrame(rows), "region_id", "cpc_id", "year", "state", validate=False
    )
    result = cluster_pair_typology_clara(
        seq,
        k=2,
        n_iterations=8,
        sample_size=12,
        random_state=42,
        verbose=False,
        aggregate_identical=False,
    )
    assert result.medoid_indices.min() >= 0
    assert result.medoid_indices.max() < seq.n_pairs
    assert len(result.unit_ids) == seq.n_pairs
    comp = result.cluster_composition()
    assert "level_1_id" in comp.columns


def test_pair_typology_global_state_colors_consistent():
    pytest.importorskip("matplotlib")
    from sequenzo.hierarchical.visualization import plot_level_portfolio_sequences

    seq = make_relational_sequences(
        _toy_long_data(), "region_id", "cpc_id", "year", "state", validate=False
    )
    import matplotlib.pyplot as plt

    ax = plot_level_portfolio_sequences(seq, level=1, max_units=2, max_sequences_per_unit=4)
    assert ax is not None
    plt.close(ax.figure)
