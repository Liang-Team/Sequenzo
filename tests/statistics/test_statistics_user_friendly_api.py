import numpy as np
import pandas as pd

from sequenzo import (
    SequenceData,
    get_distinct_state_sequences,
    get_individual_state_distribution,
    get_mean_time_by_state,
    get_modal_state_sequence,
    get_sequence_length_summary,
    get_state_spell_durations,
    get_transition_count_summary,
    get_weighted_five_number_summary,
    get_weighted_mean,
    get_weighted_variance,
)


def _toy_seqdata() -> SequenceData:
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "T1": ["A", "A"],
            "T2": ["A", "B"],
            "T3": ["B", "B"],
            "T4": ["B", "A"],
        }
    )
    return SequenceData(data=raw, time=["T1", "T2", "T3", "T4"], states=["A", "B"], id_col="id")


def test_get_distinct_state_sequences_and_spell_durations():
    s = _toy_seqdata()
    dss = get_distinct_state_sequences(s)
    dur = get_state_spell_durations(s)
    assert dss.shape[0] == s.n_sequences
    assert dur.shape[0] == s.n_sequences
    # First sequence A-A-B-B -> DSS starts with A then B -> [1,2,...]
    assert int(dss.iloc[0, 0]) == 1
    assert int(dss.iloc[0, 1]) == 2
    # First sequence durations should contain 2 and 2.
    assert int(dur.iloc[0, 0]) == 2
    assert int(dur.iloc[0, 1]) == 2


def test_get_mean_time_and_individual_distribution():
    s = _toy_seqdata()
    mean_df = get_mean_time_by_state(s, weighted=False, as_proportion=False, show_standard_error=True)
    dist_df = get_individual_state_distribution(s, as_proportion=True)
    assert "Mean" in mean_df.columns
    assert "SE" in mean_df.columns
    assert "ID" in dist_df.columns
    # Proportions should sum to 1 for each sequence.
    row_sums = dist_df.drop(columns=["ID"]).sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))

    modal_df = get_modal_state_sequence(s, weighted=False)
    assert modal_df.shape[0] == 1
    assert modal_df.shape[1] == s.n_steps

    len_summary = get_sequence_length_summary(s)
    assert {"count", "mean", "median", "q1", "q3"}.issubset(set(len_summary.columns))

    trans_summary = get_transition_count_summary(s, normalize=False)
    assert {"count", "mean", "median", "q1", "q3"}.issubset(set(trans_summary.columns))


def test_get_weighted_statistics():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 2.0, 1.0, 2.0])
    m = get_weighted_mean(x, w)
    v = get_weighted_variance(x, w)
    f = get_weighted_five_number_summary(x, w)
    assert np.isfinite(m)
    assert np.isfinite(v)
    assert f.shape == (5,)
