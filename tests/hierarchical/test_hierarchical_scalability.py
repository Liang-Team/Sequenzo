"""Scalable modes, CLARA wrapper, compression, and structural sampling."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_clara_public_api_expands_compressed_fuzzy_membership(monkeypatch):
    import sequenzo.hierarchical.clustering.clara as clara_module

    rows = []
    pair_sequences = [
        ("R1", "C1", [0, 0, 0]),
        ("R1", "C2", [1, 1, 1]),
        ("R2", "C1", [0, 0, 0]),
        ("R2", "C2", [2, 2, 2]),
        ("R3", "C1", [1, 1, 1]),
    ]
    for region, cpc, seq_values in pair_sequences:
        for t, state in enumerate(seq_values):
            rows.append({
                "region_id": region,
                "cpc_id": cpc,
                "year": 2000 + t,
                "state": state,
            })
    seq = make_relational_sequences(
        pd.DataFrame(rows), "region_id", "cpc_id", "year", "state", validate=False
    )

    compressed_membership = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
        ],
        dtype=float,
    )
    compressed_distances = np.array(
        [
            [0.0, 4.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    def fake_clara(seqdata, R, kvals, sample_size, method, dist_args, criteria, stability, max_dist):
        assert len(seqdata.ids) == 3
        assert method == "fuzzy"
        return {
            "clara": {
                0: {
                    "medoids": np.array([0, 1], dtype=int),
                    "objective": 0.0,
                    "arimatrix": np.nan,
                    "evol_diss": np.array([0.0]),
                }
            },
            "clustering": pd.DataFrame({"Cluster 2": list(compressed_membership)}),
            "stats": pd.DataFrame({
                "Avg dist": [0.0],
                "PBM": [0.0],
                "DB": [0.0],
                "XB": [0.0],
                "AMS": [0.0],
                "ARI>0.8": [0.0],
                "JC>0.8": [0.0],
                "Best iter": [0.0],
            }),
        }

    monkeypatch.setattr(clara_module, "clara", fake_clara)
    monkeypatch.setattr(
        clara_module,
        "_distances_to_medoids",
        lambda seqdata, medoid_rows, dist_args: compressed_distances,
    )

    result = cluster_pair_typology_clara(
        seq,
        k=2,
        n_iterations=1,
        sample_size=5,
        clara_method="fuzzy",
        aggregate_identical=True,
        verbose=False,
    )

    np.testing.assert_array_equal(result.unit_ids, seq.pair_ids)
    np.testing.assert_allclose(
        result.membership,
        [
            compressed_membership[0],
            compressed_membership[1],
            compressed_membership[0],
            compressed_membership[2],
            compressed_membership[1],
        ],
    )
    np.testing.assert_allclose(
        result.distance_to_medoids,
        [
            compressed_distances[0],
            compressed_distances[1],
            compressed_distances[0],
            compressed_distances[2],
            compressed_distances[1],
        ],
    )
    np.testing.assert_array_equal(result.medoid_ids, np.array(["R1__C1", "R1__C2"], dtype=object))
    np.testing.assert_array_equal(result.medoid_indices, np.array([0, 1]))
    assert result.details["compressed_medoid_ids"].tolist() == ["pattern_000001", "pattern_000002"]


def test_clara_representativeness_is_not_exposed_as_membership():
    from sequenzo.hierarchical.clustering.clara import _clara_output_to_typology

    rep = np.array([[1.0, 0.25], [0.4, 0.9]], dtype=float)
    clara_result = {
        "clara": {
            0: {
                "medoids": np.array([0, 1], dtype=int),
                "objective": 0.0,
                "arimatrix": np.nan,
                "evol_diss": np.array([0.0]),
            }
        },
        "clustering": pd.DataFrame({"Cluster 2": [rep[0], rep[1]]}),
        "stats": pd.DataFrame({
            "Avg dist": [0.0],
            "PBM": [0.0],
            "DB": [0.0],
            "XB": [0.0],
            "AMS": [0.0],
            "ARI>0.8": [0.0],
            "JC>0.8": [0.0],
            "Best iter": [0.0],
        }),
    }
    pair_ids = np.array(["p1", "p2"], dtype=object)
    seqdata = SimpleNamespace(ids=pair_ids)
    original_sequence_data = SimpleNamespace(
        pair_ids=pair_ids,
        level_1_ids=np.array(["a", "a"], dtype=object),
        level_2_ids=np.array(["b", "c"], dtype=object),
    )

    result = _clara_output_to_typology(
        clara_result,
        k=2,
        seqdata=seqdata,
        dist_args={},
        clara_method="representativeness",
        sample_size=2,
        n_iterations=1,
        level_1_ids=original_sequence_data.level_1_ids,
        level_2_ids=original_sequence_data.level_2_ids,
        weights=None,
        compression=None,
        original_sequence_data=original_sequence_data,
    )

    assert result.membership is None
    np.testing.assert_allclose(result.representativeness, rep)
    assert not np.allclose(rep.sum(axis=1), 1.0)


@pytest.mark.parametrize("clara_method", ["fuzzy", "noise"])
def test_clara_matrix_membership_expands_to_original_pairs_under_compression(monkeypatch, clara_method):
    import sequenzo.hierarchical.clustering.clara as clara_module

    membership = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    compressed_distances = np.array([[0.0, 5.0], [3.0, 0.0]], dtype=float)
    clara_result = {
        "clara": {
            0: {
                "medoids": np.array([0, 1], dtype=int),
                "objective": 0.0,
                "arimatrix": np.nan,
                "evol_diss": np.array([0.0]),
            }
        },
        "clustering": pd.DataFrame({"Cluster 2": [membership[0], membership[1]]}),
        "stats": pd.DataFrame({
            "Avg dist": [0.0],
            "PBM": [0.0],
            "DB": [0.0],
            "XB": [0.0],
            "AMS": [0.0],
            "ARI>0.8": [0.0],
            "JC>0.8": [0.0],
            "Best iter": [0.0],
        }),
    }
    compressed_records = [
        SimpleNamespace(pair_id="pattern_000001", sequence=[0, 0]),
        SimpleNamespace(pair_id="pattern_000002", sequence=[1, 1]),
    ]
    original_records = [
        SimpleNamespace(pair_id="p1", sequence=[0, 0]),
        SimpleNamespace(pair_id="p2", sequence=[1, 1]),
        SimpleNamespace(pair_id="p3", sequence=[0, 0]),
    ]
    compression = SimpleNamespace(
        compressed_data=SimpleNamespace(records=compressed_records),
        details={"n_unique_patterns": 2},
        pattern_id_to_representative_pair={
            "pattern_000001": "p1",
            "pattern_000002": "p2",
        },
    )
    seqdata = SimpleNamespace(ids=np.array(["pattern_000001", "pattern_000002"], dtype=object))
    original_sequence_data = SimpleNamespace(
        pair_ids=np.array(["p1", "p2", "p3"], dtype=object),
        level_1_ids=np.array(["a", "a", "b"], dtype=object),
        level_2_ids=np.array(["x", "y", "z"], dtype=object),
        records=original_records,
    )
    monkeypatch.setattr(
        clara_module,
        "_distances_to_medoids",
        lambda seqdata, medoid_rows, dist_args: compressed_distances,
    )

    result = clara_module._clara_output_to_typology(
        clara_result,
        k=2,
        seqdata=seqdata,
        dist_args={},
        clara_method=clara_method,
        sample_size=2,
        n_iterations=1,
        level_1_ids=original_sequence_data.level_1_ids,
        level_2_ids=original_sequence_data.level_2_ids,
        weights=np.array([2.0, 1.0]),
        compression=compression,
        original_sequence_data=original_sequence_data,
    )

    np.testing.assert_allclose(result.membership, [membership[0], membership[1], membership[0]])
    np.testing.assert_allclose(
        result.distance_to_medoids,
        [compressed_distances[0], compressed_distances[1], compressed_distances[0]],
    )
    np.testing.assert_array_equal(result.medoid_indices, np.array([0, 1]))
    np.testing.assert_array_equal(result.medoid_ids, np.array(["p1", "p2"], dtype=object))


def test_clara_representativeness_distance_contract_under_compression():
    from sequenzo.hierarchical.clustering.clara import _clara_output_to_typology

    rep = np.array([[1.0, 0.25], [0.4, 0.9]], dtype=float)
    clara_result = {
        "clara": {
            0: {
                "medoids": np.array([0, 1], dtype=int),
                "objective": 0.0,
                "arimatrix": np.nan,
                "evol_diss": np.array([0.0]),
            }
        },
        "clustering": pd.DataFrame({"Cluster 2": [rep[0], rep[1]]}),
        "stats": pd.DataFrame({
            "Avg dist": [0.0],
            "PBM": [0.0],
            "DB": [0.0],
            "XB": [0.0],
            "AMS": [0.0],
            "ARI>0.8": [0.0],
            "JC>0.8": [0.0],
            "Best iter": [0.0],
        }),
    }
    compressed_records = [
        SimpleNamespace(pair_id="pattern_000001", sequence=[0, 0]),
        SimpleNamespace(pair_id="pattern_000002", sequence=[1, 1]),
    ]
    original_records = [
        SimpleNamespace(pair_id="p1", sequence=[0, 0]),
        SimpleNamespace(pair_id="p2", sequence=[1, 1]),
        SimpleNamespace(pair_id="p3", sequence=[0, 0]),
    ]
    compression = SimpleNamespace(
        compressed_data=SimpleNamespace(records=compressed_records),
        details={"n_unique_patterns": 2},
        pattern_id_to_representative_pair={
            "pattern_000001": "p1",
            "pattern_000002": "p2",
        },
    )
    original_sequence_data = SimpleNamespace(
        pair_ids=np.array(["p1", "p2", "p3"], dtype=object),
        level_1_ids=np.array(["a", "a", "b"], dtype=object),
        level_2_ids=np.array(["x", "y", "z"], dtype=object),
        records=original_records,
    )

    result = _clara_output_to_typology(
        clara_result,
        k=2,
        seqdata=SimpleNamespace(ids=np.array(["pattern_000001", "pattern_000002"], dtype=object)),
        dist_args={},
        clara_method="representativeness",
        sample_size=2,
        n_iterations=1,
        level_1_ids=original_sequence_data.level_1_ids,
        level_2_ids=original_sequence_data.level_2_ids,
        weights=np.array([2.0, 1.0]),
        compression=compression,
        original_sequence_data=original_sequence_data,
        representativeness_max_dist=10.0,
    )

    expected_rep = np.array([rep[0], rep[1], rep[0]], dtype=float)
    np.testing.assert_allclose(result.representativeness, expected_rep)
    np.testing.assert_allclose(result.distance_to_medoids, (1.0 - expected_rep) * 10.0)
    np.testing.assert_array_equal(result.medoid_indices, np.array([0, 1]))
    np.testing.assert_array_equal(result.medoid_ids, np.array(["p1", "p2"], dtype=object))


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
