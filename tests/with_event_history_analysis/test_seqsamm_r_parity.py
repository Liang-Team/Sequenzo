"""Parity checks for TraMineRextras ``seqsamm`` (loop bound, void filter, spell.time).

For full numerical parity with R on a given dataset, also compare ``nrow(samm)``,
``range(time)``, and ``summary(spell.time)`` in R vs ``len(samm.data)`` and
``spell.time`` in Python after aligning ``SequenceData(void=...)`` and ``%`` in
``states``. See ``sequenzo/with_event_history_analysis/samm_examples.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import sequence_analysis_multi_state_model, seqsammeha
from sequenzo.with_event_history_analysis.sequence_analysis_multi_state_model import (
    SPELL_TIME_COL,
)


def _toy_seqdata(n_individuals: int = 3, n_time: int = 5) -> SequenceData:
    cols = [f"t{i}" for i in range(n_time)]
    rows = []
    for i in range(n_individuals):
        rows.append([1 + ((j + i) % 3) for j in range(n_time)])
    df = pd.DataFrame(rows, columns=cols)
    df["id"] = np.arange(1, n_individuals + 1)
    return SequenceData(
        df,
        time=cols,
        id_col="id",
        states=[1, 2, 3],
        labels=["A", "B", "C"],
    )


def test_seqsamm_censoring_time_limit():
    """R: tt in 1:(L - sublength) => max(time) == L - sublength."""
    seq = _toy_seqdata(n_time=5)
    sublength = 2
    samm = sequence_analysis_multi_state_model(seq, sublength=sublength)
    assert int(samm.data["time"].max()) == 5 - sublength


def test_seqsamm_spell_time_column():
    seq = _toy_seqdata()
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    assert SPELL_TIME_COL in samm.data.columns
    assert "spell_time" in samm.data.columns
    assert (samm.data["spell_time"] == samm.data[SPELL_TIME_COL]).all()


def test_seqsamm_subsequence_columns_are_codes():
    seq = _toy_seqdata()
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    for col in samm.sname:
        assert samm.data[col].isin([1, 2, 3]).all()


def test_seqsamm_covar_indexed_by_id():
    seq = _toy_seqdata()
    covar = pd.DataFrame({"x": [10, 20, 30]}, index=[1, 2, 3])
    samm = sequence_analysis_multi_state_model(seq, sublength=2, covar=covar)
    assert "x" in samm.data.columns
    assert samm.data.loc[samm.data["id"] == 1, "x"].iloc[0] == 10


def test_seqsamm_covar_index_dtype_mismatch_raises():
    seq = _toy_seqdata()
    covar = pd.DataFrame({"x": [10, 20, 30]}, index=["1", "2", "3"])
    with pytest.raises(ValueError, match="dtypes differ"):
        sequence_analysis_multi_state_model(seq, sublength=2, covar=covar)


def test_seqsammeha_typology_factor_levels():
    seq = _toy_seqdata()
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    spell_code = 1
    n_trans = int(((samm.data["s.1"] == spell_code) & samm.data["transition"]).sum())
    typology = pd.Categorical(["a"] * n_trans, categories=["a", "b", "c"])
    eha = seqsammeha(samm, spell=spell_code, typology=typology, persper=True)
    assert "SAMMa" in eha.columns
    assert "SAMMb" in eha.columns
    assert "SAMMc" in eha.columns
    assert (eha["SAMMc"] == 0).all()
