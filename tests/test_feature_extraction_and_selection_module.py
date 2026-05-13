"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_feature_extraction_and_selection_module.py
@Time    : 23/03/2026 09:41
@Desc    :
    Pytest smoke tests for the Feature Extraction & Selection (FES) submodule.

    The goal is to validate that:
    - the readable pipeline API can run end-to-end on small monthly state data
    - both regression and classification modes work
    - the clustassoc-like validation utility returns a correctly-shaped table
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from sequenzo import SequenceData
from sequenzo import (
    run_feature_extraction_and_selection_pipeline,
    FeatureExtractionAndSelectionConfig,
    clustassoc_like_typology_validation,
    cluster_correlated_features,
)
from sequenzo.feature_extraction_and_selection.time_binning_utils import (
    suggest_timing_bin_width,
)
from sequenzo.feature_extraction_and_selection.feature_extraction import (
    extract_sequence_features,
)
from sequenzo.feature_extraction_and_selection.clustassoc_typology_validation import (
    _compute_pseudo_r2_for_terms,
    _design_with_intercept,
    _one_hot,
)


def _build_small_sequence_data():
    """Three individuals, five time points, two states."""
    time_cols = [0, 1, 2, 3, 4]
    ids = [1, 2, 3]
    seqs = [
        ["A", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "B", "B"],
    ]
    rows = []
    for uid, s in zip(ids, seqs):
        row = {"id": uid}
        for t, v in zip(time_cols, s):
            row[t] = v
        rows.append(row)
    return SequenceData(pd.DataFrame(rows), time=time_cols, states=["A", "B"], id_col="id")


def _build_toy_monthly_sequence_data(*, n_individuals: int = 30):
    """
    Build monthly state data where outcome can correlate with spell features.

    Each individual has a monotone spell pattern: A repeated k times, then B.
    Default n=30 is large enough for BorutaPy to confirm features on toy runs.
    """
    time_cols = list(range(8))
    ids = list(range(1, n_individuals + 1))
    rows = []
    for uid in ids:
        n_a = uid % (len(time_cols) + 1)
        seq = ["A"] * n_a + ["B"] * (len(time_cols) - n_a)
        row = {"id": uid}
        for t, v in zip(time_cols, seq):
            row[t] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    seqdata = SequenceData(df, time=time_cols, states=["A", "B"], id_col="id")
    return seqdata


def test_fes_pipeline_regression_smoke(capsys):
    """
    Regression smoke test:
    - run the full FES pipeline
    - verify at least one feature is selected
    - verify final model metrics exist
    """
    seqdata = _build_toy_monthly_sequence_data()

    # Suppress SequenceData printing (optional).
    capsys.readouterr()

    y = np.array([(uid % (len(seqdata.time) + 1)) for uid in seqdata.ids], dtype=float)

    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=30,
        boruta_perc=100.0,
        residualize_target_with_controls=False,
    )

    res = run_feature_extraction_and_selection_pipeline(
        seqdata=seqdata,
        outcome=y,
        problem_type="regression",
        controls=None,
        config=cfg,
        ids=seqdata.ids.tolist(),
        fit_final_model=True,
        verbose=False,
    )

    assert res["problem_type"] == "regression"
    assert len(res["selected_feature_names"]) >= 1
    assert res["final_model_fitted"] is True
    assert res["final_model_is_exploratory"] is True
    assert "final_model" in res
    assert np.isfinite(res["r2"])
    assert res["X_selected"].shape[0] == len(y)


def test_fes_pipeline_classification_smoke(capsys):
    """
    Classification smoke test:
    - run the full FES pipeline in classification mode
    - verify at least one feature is selected
    - verify accuracy is returned and is within [0,1]
    """
    seqdata = _build_toy_monthly_sequence_data()
    capsys.readouterr()

    y = np.array([1 if (uid % (len(seqdata.time) + 1)) >= 4 else 0 for uid in seqdata.ids], dtype=int)

    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=30,
        boruta_perc=100.0,
        residualize_target_with_controls=False,
    )

    res = run_feature_extraction_and_selection_pipeline(
        seqdata=seqdata,
        outcome=y,
        problem_type="classification",
        controls=None,
        config=cfg,
        ids=seqdata.ids.tolist(),
        fit_final_model=True,
        verbose=False,
    )

    assert res["problem_type"] == "classification"
    assert len(res["selected_feature_names"]) >= 1
    assert res["final_model_fitted"] is True
    assert "final_model" in res
    assert np.isfinite(res["accuracy"])
    assert 0.0 <= res["accuracy"] <= 1.0


def test_clustassoc_like_typology_validation_smoke():
    """
    Smoke test for the clustassoc-like validation table.

    We generate a symmetric dissimilarity matrix and provide clustering labels
    for two different k values. We then check that the returned table:
    - has two rows
    - contains the expected columns
    - has finite pseudo-R2 numbers
    """
    rng = np.random.default_rng(0)
    n = 8
    A = rng.random((n, n))
    diss = (A + A.T) / 2.0
    np.fill_diagonal(diss, 0.0)

    covar = rng.normal(size=n)
    labels_by_k = {
        2: rng.integers(0, 2, size=n),
        3: rng.integers(0, 3, size=n),
    }

    table = clustassoc_like_typology_validation(
        diss=diss,
        covariate=covar,
        clustering_labels_by_k=labels_by_k,
        covariate_is_categorical=False,
        verbose=False,
    )

    assert table.shape[0] == 2
    expected_cols = {
        "k",
        "pseudoR2_original",
        "pseudoR2_remaining_after_clustering",
        "association_unaccounted_share",
        "association_accounted_share",
    }
    assert expected_cols.issubset(set(table.columns))
    assert np.all(np.isfinite(table["pseudoR2_original"].to_numpy()))
    assert np.all(np.isfinite(table["pseudoR2_remaining_after_clustering"].to_numpy()))


def test_suggest_timing_bin_width_accepts_positional_arg():
    assert suggest_timing_bin_width("month") == 12.0
    assert suggest_timing_bin_width("year") == 1.0


def test_pipeline_without_final_model_flags():
    seqdata = _build_toy_monthly_sequence_data()
    y = np.array([(uid % (len(seqdata.time) + 1)) for uid in seqdata.ids], dtype=float)
    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=30,
        residualize_target_with_controls=False,
    )
    res = run_feature_extraction_and_selection_pipeline(
        seqdata=seqdata,
        outcome=y,
        problem_type="regression",
        config=cfg,
        ids=seqdata.ids.tolist(),
        fit_final_model=False,
        verbose=False,
    )
    assert res["final_model_fitted"] is False
    assert res["final_model_is_exploratory"] is False
    assert "final_model" not in res


def test_multiclass_residualization_raises():
    seqdata = _build_toy_monthly_sequence_data(n_individuals=12)
    y = np.array([0, 1, 2] * 4, dtype=int)
    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=5,
        residualize_target_with_controls=True,
    )
    controls = np.column_stack([np.ones(len(y)), np.arange(len(y), dtype=float)])
    with pytest.raises(ValueError, match="binary outcomes"):
        run_feature_extraction_and_selection_pipeline(
            seqdata=seqdata,
            outcome=y,
            problem_type="classification",
            controls=controls,
            config=cfg,
            ids=seqdata.ids.tolist(),
            verbose=False,
        )


def test_extract_sequence_features_rejects_weighted_sequencing():
    seqdata = _build_toy_monthly_sequence_data(n_individuals=6)
    with pytest.raises(NotImplementedError, match="Weighted sequencing"):
        extract_sequence_features(seqdata, sequencing_weighted=True)


def test_cluster_correlated_features_hierarchical_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = a + rng.normal(scale=0.05, size=20)
    c = rng.normal(size=20)
    X = np.column_stack([a, b, c])
    out = cluster_correlated_features(X, ["A", "B", "C"], abs_corr_threshold=0.7)
    cid_a = int(out.loc[out["feature"] == "A", "cluster_id"].iloc[0])
    cid_b = int(out.loc[out["feature"] == "B", "cluster_id"].iloc[0])
    cid_c = int(out.loc[out["feature"] == "C", "cluster_id"].iloc[0])
    assert cid_a == cid_b
    assert cid_c != cid_a


def test_extract_sequence_features_small_panel(capsys):
    seqdata = _build_small_sequence_data()
    capsys.readouterr()

    out = extract_sequence_features(
        seqdata,
        timing_bin_width=2.0,
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        ids=seqdata.ids.tolist(),
    )

    assert out["X_duration"].shape[0] == 3
    assert out["X_timing"].shape[0] == 3
    assert out["X_sequencing"].shape[0] == 3
    assert out["X_full"].shape[0] == 3
    assert len(out["all_feature_names"]) == out["X_full"].shape[1]
    assert any(name.startswith("DUR_") for name in out["all_feature_names"])
    assert any("START_" in name or "END_" in name for name in out["all_feature_names"])


def test_clustassoc_term_labels_not_misaligned():
    rng = np.random.default_rng(1)
    n = 10
    diss = rng.random((n, n))
    diss = (diss + diss.T) / 2.0
    np.fill_diagonal(diss, 0.0)
    covar = rng.normal(size=n)
    labels = rng.integers(0, 2, size=n)

    design_null = _design_with_intercept(covar.reshape(-1, 1))
    summary_null = _compute_pseudo_r2_for_terms(
        diss=diss,
        design=design_null,
        term_ids=[0, 1],
        term_labels=["Covariate"],
    )
    assert "Covariate" in summary_null["Variable"].tolist()
    assert "Intercept" not in summary_null["Variable"].tolist()

    X_clust = _one_hot(labels, drop_first=True)
    design_full = _design_with_intercept(np.hstack([X_clust, covar.reshape(-1, 1)]))
    summary_full = _compute_pseudo_r2_for_terms(
        diss=diss,
        design=design_full,
        term_ids=[0] + [1] * X_clust.shape[1] + [2],
        term_labels=["Clustering", "Covariate"],
    )
    variables = summary_full["Variable"].tolist()
    assert "Clustering" in variables
    assert "Covariate" in variables
    assert variables.index("Clustering") < variables.index("Covariate")


def test_binary_classification_residualization_runs(capsys):
    seqdata = _build_toy_monthly_sequence_data()
    capsys.readouterr()

    y = np.array([1 if (uid % (len(seqdata.time) + 1)) >= 4 else 0 for uid in seqdata.ids], dtype=int)
    controls = np.column_stack(
        [
            np.ones(len(y)),
            np.arange(len(y), dtype=float),
        ]
    )
    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.3,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=30,
        residualize_target_with_controls=True,
    )
    res = run_feature_extraction_and_selection_pipeline(
        seqdata=seqdata,
        outcome=y,
        problem_type="classification",
        controls=controls,
        config=cfg,
        ids=seqdata.ids.tolist(),
        fit_final_model=False,
        verbose=False,
    )
    assert res["problem_type"] == "classification"
    assert len(res["selected_feature_names"]) >= 1


