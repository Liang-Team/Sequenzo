"""
Numerical parity tests for property-based clustering vs WeightedCluster ``seqpropclust``.

Generate reference CSV files with::

    Rscript tests/clustering/weightedcluster_reference_seqpropclust.R tests/clustering
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.clustering.property_based_clustering import (
    cut_tree,
    extract_sequence_properties,
    seqpropclust,
)
from sequenzo.datasets import load_dataset
from sequenzo.dissimilarity_measures import get_distance_matrix

RTOL = 1e-5
ATOL = 1e-6
REF_DIR = Path(__file__).resolve().parent


def _load_ref(name: str) -> pd.DataFrame | None:
    path = REF_DIR / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _tiny_seqdata() -> SequenceData:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "1": [1, 1, 2],
            "2": [1, 2, 2],
            "3": [2, 2, 3],
            "4": [2, 3, 3],
        }
    )
    return SequenceData(
        df,
        time=["1", "2", "3", "4"],
        id_col="id",
        states=[1, 2, 3],
        labels=["A", "B", "C"],
    )


def _lsog_seqdata() -> SequenceData:
    df = load_dataset("dyadic_children").head(20)
    time_list = sorted([c for c in df.columns if str(c).isdigit()], key=int)
    return SequenceData(df, time=time_list, id_col="dyadID", states=[1, 2, 3, 4, 5, 6])


def _compare_property_frames(py: pd.DataFrame, ref: pd.DataFrame) -> None:
    assert list(py.columns) == list(ref.columns)
    for col in py.columns:
        actual = py[col]
        expected = ref[col]
        if pd.api.types.is_numeric_dtype(actual) and pd.api.types.is_numeric_dtype(expected):
            np.testing.assert_allclose(
                actual.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=RTOL,
                atol=ATOL,
                err_msg=col,
            )
        else:
            assert actual.astype(str).tolist() == expected.astype(str).tolist()


@pytest.fixture
def tiny_seqdata():
    return _tiny_seqdata()


@pytest.fixture
def lsog_seqdata():
    return _lsog_seqdata()


def test_spell_properties_use_numeric_dss_codes(tiny_seqdata):
    props = extract_sequence_properties(
        tiny_seqdata,
        properties=("spell.dur", "spell.age"),
        verbose=False,
    )
    assert "spell.A_dur_1" in props.columns
    assert props.loc[0, "spell.A_dur_1"] == 2.0
    assert props.loc[0, "spell.A_age_1"] == 0.0
    assert props.loc[0, "spell.B_age_1"] == 2.0


@pytest.mark.parametrize(
    ("ref_name", "properties"),
    [
        ("ref_seqpropclust_tiny_all.csv", ("state", "duration", "spell.age", "spell.dur")),
        (
            "ref_seqpropclust_lsog_core.csv",
            ("state", "duration", "spell.age", "spell.dur", "Complexity"),
        ),
    ],
)
def test_extract_sequence_properties_matches_weightedcluster(ref_name, properties, tiny_seqdata, lsog_seqdata):
    ref = _load_ref(ref_name)
    if ref is None:
        pytest.skip(f"Reference file {ref_name} not found. Run weightedcluster_reference_seqpropclust.R.")

    seqdata = tiny_seqdata if "tiny" in ref_name else lsog_seqdata
    py = extract_sequence_properties(seqdata, properties=properties, verbose=False)
    _compare_property_frames(py, ref)


@pytest.mark.parametrize(
    ("ref_name", "properties"),
    [
        ("ref_seqpropclust_tiny_state.csv", ("state",)),
        ("ref_seqpropclust_tiny_duration.csv", ("duration",)),
        ("ref_seqpropclust_tiny_spell.csv", ("spell.age", "spell.dur")),
    ],
)
def test_single_property_blocks_match_weightedcluster(ref_name, properties, tiny_seqdata):
    ref = _load_ref(ref_name)
    if ref is None:
        pytest.skip(f"Reference file {ref_name} not found.")

    py = extract_sequence_properties(tiny_seqdata, properties=properties, verbose=False)
    py = py.rename(columns=lambda col: col.split(".", 1)[-1] if "." in col else col)
    _compare_property_frames(py, ref)


def test_seqpropclust_cuts_match_weightedcluster_for_selected_k(lsog_seqdata):
    ref = _load_ref("ref_seqpropclust_lsog_cuts.csv")
    if ref is None:
        pytest.skip("Reference cuts file not found.")

    diss = get_distance_matrix(seqdata=lsog_seqdata, method="LCS", norm="auto")
    if isinstance(diss, pd.DataFrame):
        diss = diss.to_numpy()

    tree = seqpropclust(
        lsog_seqdata,
        diss=diss,
        properties=["state", "duration"],
        max_clusters=6,
        R=1,
        weight_permutation="diss",
        min_size=0.01,
        max_depth=5,
        pval=1.0,
        verbose=False,
    )

    for k in (2, 3, 4, 6):
        actual = cut_tree(tree, n_clusters=k, labels=False)
        expected = ref[f"Split{k}"].astype(int).to_numpy()
        np.testing.assert_array_equal(actual, expected)


def test_seqpropclust_fitted_leaves_match_weightedcluster(lsog_seqdata):
    ref = _load_ref("ref_seqpropclust_lsog_fitted.csv")
    if ref is None:
        pytest.skip("Reference fitted file not found.")

    diss = get_distance_matrix(seqdata=lsog_seqdata, method="LCS", norm="auto")
    if isinstance(diss, pd.DataFrame):
        diss = diss.to_numpy()

    tree = seqpropclust(
        lsog_seqdata,
        diss=diss,
        properties=["state", "duration"],
        max_clusters=6,
        R=1,
        weight_permutation="diss",
        min_size=0.01,
        max_depth=5,
        pval=1.0,
        verbose=False,
    )
    fitted = tree["fitted"]["(fitted)"].to_numpy()
    by_id = dict(zip(load_dataset("dyadic_children").head(20)["dyadID"], fitted))
    actual = np.array([by_id[row_id] for row_id in ref["id"]], dtype=int)
    np.testing.assert_array_equal(actual, ref["fitted"].to_numpy(dtype=int))
