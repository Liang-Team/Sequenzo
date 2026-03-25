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
)
from sequenzo import run_fes_pipeline, FESConfig


def _build_toy_monthly_sequence_data():
    """
    Build a tiny monthly state dataset with 2 states and a few individuals.

    We use a constant grid of 5 time points: [0,1,2,3,4].
    """
    time_cols = [0, 1, 2, 3, 4]
    ids = [1, 2, 3, 4, 5, 6]
    seqs = [
        ["A", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "B"],
        ["A", "B", "A", "B", "B"],
    ]

    rows = []
    for uid, s in zip(ids, seqs):
        row = {"id": uid}
        for t, v in zip(time_cols, s):
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

    y = np.array([10.0, 12.0, 11.0, 7.0, 9.0, 8.0], dtype=float)

    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.5,  # high support to keep subseq mining small
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=2,  # keep tests fast
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
        verbose=False,
    )

    assert res["problem_type"] == "regression"
    assert len(res["selected_feature_names"]) >= 1
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

    # Binary labels
    y = np.array([1, 1, 1, 0, 1, 0], dtype=int)

    cfg = FeatureExtractionAndSelectionConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.5,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=2,  # keep tests fast
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
        verbose=False,
    )

    assert res["problem_type"] == "classification"
    assert len(res["selected_feature_names"]) >= 1
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
        "pseudoR2_null",
        "pseudoR2_remaining",
        "unaccounted_share",
        "accounted_share",
    }
    assert expected_cols.issubset(set(table.columns))
    assert np.all(np.isfinite(table["pseudoR2_null"].to_numpy()))
    assert np.all(np.isfinite(table["pseudoR2_remaining"].to_numpy()))


def test_backward_compat_aliases_smoke(capsys):
    """
    Backward compatibility smoke test:
    - the old alias names should still be importable and callable
    """
    seqdata = _build_toy_monthly_sequence_data()
    capsys.readouterr()

    y = np.array([10.0, 12.0, 11.0, 7.0, 9.0, 8.0], dtype=float)

    cfg = FESConfig(
        sequencing_max_k=2,
        sequencing_min_support=0.5,
        sequencing_top_mined_subsequences=10,
        timing_bin_width=2.0,
        boruta_n_iter=2,
        residualize_target_with_controls=False,
    )

    res = run_fes_pipeline(
        seqdata=seqdata,
        y=y,
        problem_type="regression",
        controls=None,
        config=cfg,
        ids=seqdata.ids.tolist(),
        verbose=False,
    )

    assert res["problem_type"] == "regression"
    assert len(res["selected_feature_names"]) >= 1

