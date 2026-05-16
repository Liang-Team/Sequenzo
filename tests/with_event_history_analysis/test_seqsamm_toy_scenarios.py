"""
Three toy ``seqsamm`` scenarios for regression checks (no void, with void, with covar).

Golden values match TraMineRextras ``seqsamm()`` on the same data when run with
``tests/with_event_history_analysis/seqsamm_toy_reference.R``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import (
    sequence_analysis_multi_state_model,
    seqsammeha,
)
from sequenzo.with_event_history_analysis.sequence_analysis_multi_state_model import (
    SPELL_TIME_COL,
)


def _toy_no_void_seq() -> SequenceData:
    cols = [f"t{i}" for i in range(5)]
    rows = [[1 + ((j + i) % 3) for j in range(5)] for i in range(3)]
    df = pd.DataFrame(rows, columns=cols)
    df["id"] = [1, 2, 3]
    return SequenceData(
        df,
        time=cols,
        id_col="id",
        states=[1, 2, 3],
        labels=["A", "B", "C"],
        void=None,
    )


def _toy_with_void_seq() -> SequenceData:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "t1": ["%", "%"],
            "t2": ["%", "A"],
            "t3": ["A", "A"],
            "t4": ["B", "B"],
        }
    )
    return SequenceData(
        df,
        time=["t1", "t2", "t3", "t4"],
        id_col="id",
        states=["%", "A", "B"],
        void="%",
    )


def test_seqsamm_sublength_validation():
    seq = _toy_no_void_seq()
    with pytest.raises(ValueError, match="at least 2"):
        sequence_analysis_multi_state_model(seq, sublength=1)
    with pytest.raises(ValueError, match="greater than sublength"):
        sequence_analysis_multi_state_model(seq, sublength=5)


def test_seqsamm_toy_no_void():
    """Complete sequences, void=None: 3 ids × 3 time positions, all transitions."""
    samm = sequence_analysis_multi_state_model(_toy_no_void_seq(), sublength=2)
    d = samm.data

    assert len(d) == 9
    assert int(d["time"].min()) == 1
    assert int(d["time"].max()) == 3
    assert set(d.columns) >= {"id", "time", SPELL_TIME_COL, "transition", "s.1", "s.2"}
    assert d["transition"].all()
    assert (d["s.1"] != d["s.2"]).all()
    assert d["s.1"].isin([1, 2, 3]).all() and d["s.2"].isin([1, 2, 3]).all()


def test_seqsamm_toy_with_void():
    """Void rows dropped: only id=2 at time=2 survives (subseq [A, A], no void)."""
    seq = _toy_with_void_seq()
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    d = samm.data
    void_code = seq.void_code

    assert len(d) == 1
    assert d["id"].iloc[0] == 2
    assert int(d["time"].iloc[0]) == 2
    for col in samm.sname:
        assert (d[col] != void_code).all()


def test_seqsamm_toy_with_covar():
    """Covar merged by id (R: covar[ret$id, ])."""
    seq = _toy_no_void_seq()
    covar = pd.DataFrame({"x": [10, 20, 30]}, index=[1, 2, 3])
    samm = sequence_analysis_multi_state_model(seq, sublength=2, covar=covar)
    d = samm.data

    assert len(d) == 9
    assert d.groupby("id")["x"].first().tolist() == [10, 20, 30]
    assert not d["x"].isna().any()


def test_seqsammeha_typology_categorical_series_levels():
    seq = _toy_no_void_seq()
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    spell_code = 1
    n_trans = int(((samm.data["s.1"] == spell_code) & samm.data["transition"]).sum())
    typology = pd.Series(
        pd.Categorical(["a"] * n_trans, categories=["a", "b", "c"])
    )
    eha = seqsammeha(samm, spell=spell_code, typology=typology, persper=True)
    assert "SAMMa" in eha.columns
    assert "SAMMb" in eha.columns
    assert "SAMMc" in eha.columns
    assert (eha["SAMMc"] == 0).all()
