"""Parity tests: R TraMineRextras ``seqsurv`` vs ``get_spell_survival_analysis`` on mvad."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, load_dataset
from sequenzo.with_event_history_analysis import get_spell_survival_analysis

REF_DIR = Path(__file__).resolve().parent / "reference_data"
RTOL = 0
ATOL = 1e-9


def _mvad_sequence_data() -> SequenceData:
    df = load_dataset("mvad")
    time_cols = list(df.columns[16:86])
    alphabet = ["employment", "FE", "HE", "joblessness", "school", "training"]
    labels = [
        "employment",
        "further education",
        "higher education",
        "joblessness",
        "school",
        "training",
    ]
    return SequenceData(
        df,
        time=time_cols,
        id_col="id",
        states=alphabet,
        labels=labels,
    )


@pytest.fixture(scope="module")
def r_reference():
    path = REF_DIR / "ref_mvad_seqsurv.csv"
    if not path.exists():
        pytest.skip(
            "R reference missing. Run: "
            "Rscript tests/with_event_history_analysis/traminerextras_reference_seqsurv_mvad.R"
        )
    return pd.read_csv(path)


def test_get_spell_survival_analysis_mvad_matches_r_seqsurv(r_reference):
    seq = _mvad_sequence_data()
    fit = get_spell_survival_analysis(seq)
    py = fit.to_summary_frame()

    for state in seq.states:
        p = py[py["strata"] == state].sort_values("time")
        r = r_reference[r_reference["strata"] == state].sort_values("time")
        assert len(p) == len(r), state
        np.testing.assert_allclose(p["surv"], r["surv"], rtol=RTOL, atol=ATOL, err_msg=state)
        np.testing.assert_allclose(p["n.risk"], r["n.risk"], rtol=RTOL, atol=ATOL, err_msg=state)
        np.testing.assert_allclose(p["n.event"], r["n.event"], rtol=RTOL, atol=ATOL, err_msg=state)


def test_get_spell_survival_analysis_rejects_unobserved_state():
    seq = _mvad_sequence_data()
    with pytest.raises(ValueError, match="unobserved states"):
        get_spell_survival_analysis(seq, state=["nonexistent_state"])


def test_get_spell_survival_analysis_rejects_multiple_groups_without_per_state():
    seq = _mvad_sequence_data()
    groups = ["A", "B"] * (len(seq.values) // 2 + 1)
    groups = groups[: len(seq.values)]
    with pytest.raises(ValueError, match="single group"):
        get_spell_survival_analysis(seq, groups=groups, per_state=False)
