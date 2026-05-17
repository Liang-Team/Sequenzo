"""
MVP tests for sequenzo.hierarchical (relational / hierarchical sequence analysis).
"""

import numpy as np
import pandas as pd
import pytest

from sequenzo.hierarchical import (
    validate_relational_sequence_data,
    make_relational_sequences,
    compute_relational_distance_matrix,
    summarize_distance_by_structure,
    sequence_discrepancy_by_level,
    permutation_test_level_effect,
    hierarchical_sequence_discrepancy,
    run_hierarchical_sequence_analysis,
    state_sequence_to_spells,
)


def _toy_long_data():
    """Small region–CPC style panel."""
    rows = []
    for region in ("R1", "R2"):
        for cpc in ("C1", "C2"):
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


def test_validate_and_make_sequences():
    df = _toy_long_data()
    summary = validate_relational_sequence_data(
        df, "region_id", "cpc_id", "year", "state"
    )
    assert summary["n_level_1"] == 2
    assert summary["n_level_2"] == 2
    assert summary["n_pairs"] == 4
    assert summary["balanced"] is True

    seq_data = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=True
    )
    assert seq_data.n_pairs == 4
    assert seq_data.records[0].pair_id == "R1__C1"
    assert len(seq_data.records[0].sequence) == 4


def test_spell_representation():
    spells = state_sequence_to_spells([0, 0, 1, 1, 2])
    assert spells == [(0, 2), (1, 2), (2, 1)]


def test_distance_and_structural_summary():
    df = _toy_long_data()
    seq_data = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    dist = compute_relational_distance_matrix(seq_data, method="HAM")
    assert dist.matrix.shape == (4, 4)
    assert np.allclose(dist.matrix, dist.matrix.T)
    assert len(dist.level_1_ids) == 4

    struct = summarize_distance_by_structure(dist)
    assert "mean_distance" in struct.columns
    baseline = struct.loc[
        struct["comparison_type"].str.contains("baseline"), "mean_distance"
    ].iloc[0]
    assert np.isfinite(baseline)


def test_discrepancy_by_level():
    df = _toy_long_data()
    seq_data = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    dist = compute_relational_distance_matrix(seq_data, method="HAM")

    r1 = sequence_discrepancy_by_level(dist, dist.level_1_ids, grouping_variable="region")
    assert 0 <= r1.pseudo_r2 <= 1
    assert r1.total_discrepancy >= 0

    decomp = hierarchical_sequence_discrepancy(dist)
    assert decomp.level_1.pseudo_r2 >= 0
    assert decomp.level_2.pseudo_r2 >= 0


def test_permutation_test_smoke():
    df = _toy_long_data()
    seq_data = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    dist = compute_relational_distance_matrix(seq_data, method="HAM")
    out = permutation_test_level_effect(
        dist, dist.level_1_ids, n_perm=49, random_state=1
    )
    assert "observed_pseudo_r2" in out
    assert "p_value" in out
    assert 0 <= out["p_value"] <= 1


def test_run_pipeline():
    df = _toy_long_data()
    result = run_hierarchical_sequence_analysis(
        df,
        level_1_col="region_id",
        level_2_col="cpc_id",
        time_col="year",
        state_col="state",
        distance_method="HAM",
        n_perm=19,
        random_state=0,
    )
    text = result.summary()
    assert "Hierarchical Sequence Analysis Summary" in text
    assert result.decomposition is not None
    assert len(result.level_1_profiles) == 2


def test_duplicate_pair_time_raises():
    df = _toy_long_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_relational_sequence_data(
            df, "region_id", "cpc_id", "year", "state"
        )
