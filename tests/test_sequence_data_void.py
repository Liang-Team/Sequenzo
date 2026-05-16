"""Tests for TraMineR-compatible void metadata on SequenceData."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import sequence_analysis_multi_state_model


def _df_with_void():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "t1": ["%", "%"],
            "t2": ["%", "A"],
            "t3": ["A", "A"],
            "t4": ["B", "B"],
        }
    )


def test_sequence_data_void_default_and_void_code():
    df = _df_with_void()
    seq = SequenceData(
        df,
        time=["t1", "t2", "t3", "t4"],
        id_col="id",
        states=["%", "A", "B"],
        labels=["Void", "A", "B"],
    )
    assert seq.void == "%"
    assert seq.void_code == 1
    assert seq.has_void_in_data()


def test_sequence_data_void_none_disables_code():
    df = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(
        df,
        time=["t1", "t2"],
        id_col="id",
        states=["A", "B"],
        void=None,
    )
    assert seq.void is None
    assert seq.void_code is None


def test_sequence_data_void_in_data_must_be_in_states():
    df = _df_with_void()
    with pytest.raises(ValueError, match="void symbol"):
        SequenceData(
            df,
            time=["t1", "t2", "t3", "t4"],
            id_col="id",
            states=["A", "B"],
            void="%",
        )


def test_seqsamm_drops_subsequences_with_void():
    df = _df_with_void()
    seq = SequenceData(
        df,
        time=["t1", "t2", "t3", "t4"],
        id_col="id",
        states=["%", "A", "B"],
        void="%",
    )
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    void_code = seq.void_code
    for col in samm.sname:
        assert (samm.data[col] != void_code).all()


def test_seqsamm_no_void_filter_when_void_none():
    df = _df_with_void()
    seq = SequenceData(
        df,
        time=["t1", "t2", "t3", "t4"],
        id_col="id",
        states=["%", "A", "B"],
        void=None,
    )
    samm = sequence_analysis_multi_state_model(seq, sublength=2)
    assert (samm.data["s.1"] == seq.state_mapping["%"]).any()
