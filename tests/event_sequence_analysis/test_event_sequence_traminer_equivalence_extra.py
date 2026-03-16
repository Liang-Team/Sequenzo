import os
import numpy as np
import pandas as pd
import pytest

from sequenzo.datasets import load_dataset
from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import (
    create_event_sequences,
    convert_event_sequences_to_tse,
    compute_event_transition_matrix,
    check_event_subsequence_containment,
    EventSequenceConstraint,
)


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def lsog_seqdata():
    """
    Load a small LSOG (dyadic_children) example and build SequenceData.

    This fixture mirrors the one used in test_event_sequence_lsog.py so that
    reference-based tests here can reuse the exact same data configuration
    as the TraMineR R scripts.
    """
    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    df = df.head(20)
    states = [1, 2, 3, 4, 5, 6]
    seqdata = SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )
    return seqdata


@pytest.fixture
def small_eseq_from_tse():
    """
    Build a tiny TSE example by hand.

    This fixture is deliberately simple so that the expected TSE and
    transition matrix can be worked out analytically, and compared to
    TraMineR's documented behaviour of seqe2tse() and seqetm().
    """
    data = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "timestamp": [0.0, 1.0, 3.0, 0.0, 2.0],
            "event": ["A", "B", "C", "A", "C"],
        }
    )
    eseq = create_event_sequences(data=data)
    return eseq, data


def test_convert_event_sequences_to_tse_matches_original_tse(small_eseq_from_tse):
    """
    convert_event_sequences_to_tse should reproduce the original TSE table
    (up to row ordering), just like TraMineR's seqe2tse() is the inverse
    of seqecreate() when starting from TSE.
    """
    eseq, original_tse = small_eseq_from_tse
    tse_out = convert_event_sequences_to_tse(eseq)

    # Sort both tables in a canonical way before comparison
    original_sorted = (
        original_tse.sort_values(["id", "timestamp", "event"])
        .reset_index(drop=True)
        .astype({"id": int, "timestamp": float, "event": str})
    )
    out_sorted = (
        tse_out.sort_values(["id", "timestamp", "event"])
        .reset_index(drop=True)
        .astype({"id": int, "timestamp": float, "event": str})
    )

    pd.testing.assert_frame_equal(original_sorted, out_sorted)


def test_compute_event_transition_matrix_simple_counts(small_eseq_from_tse):
    """
    compute_event_transition_matrix should count transitions exactly as
    TraMineR's seqetm(): each consecutive pair of events within a sequence
    contributes one unit (or its weight) to the (from, to) cell.

    For the hand-built example we can derive the transition counts by hand:
        id=1 : 0:A -> 1:B, 1:B -> 3:C
        id=2 : 0:A -> 2:C
    So expected (row: from, col: to) counts are:
        A->B : 1
        B->C : 1
        A->C : 1
    All other cells are zero.
    """
    eseq, _ = small_eseq_from_tse
    tm = compute_event_transition_matrix(eseq, weighted=True, normalize=False)

    # Ensure we have the expected alphabet ordering
    assert list(tm.index) == eseq.dictionary
    assert list(tm.columns) == eseq.dictionary

    # Build an explicit expected matrix in the same label order
    labels = eseq.dictionary
    idx = {lab: i for i, lab in enumerate(labels)}
    expected = np.zeros_like(tm.values, dtype=float)
    expected[idx["A"], idx["B"]] = 1.0
    expected[idx["B"], idx["C"]] = 1.0
    expected[idx["A"], idx["C"]] = 1.0

    np.testing.assert_array_equal(tm.values, expected)


def test_compute_event_transition_matrix_row_normalization(small_eseq_from_tse):
    """
    With normalize=True, each row of the transition matrix should sum to 1
    when there is at least one outgoing transition, matching TraMineR's
    probability-normalised event transition matrix.
    """
    eseq, _ = small_eseq_from_tse
    tm = compute_event_transition_matrix(eseq, weighted=True, normalize=True)

    row_sums = tm.sum(axis=1).values

    # For rows that have at least one outgoing transition, the sum must be 1
    has_outgoing = row_sums > 0
    np.testing.assert_allclose(row_sums[has_outgoing], np.ones(has_outgoing.sum()))


def test_check_event_subsequence_containment_basic():
    """
    check_event_subsequence_containment should agree with the internal
    subsequence search logic and, by design, TraMineR's seqecontain().

    We construct two sequences:
        s1: (A)-(B)-(C)
        s2: (A)-(C)
    and ask whether the subsequence "(A)-(B)" appears.
    Only the first sequence should return True.
    """
    # Build a very small EventSequenceList by hand through create_event_sequences
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "timestamp": [0.0, 1.0, 0.0, 1.0],
            "event": ["A", "B", "A", "C"],
        }
    )
    eseq = create_event_sequences(data=data)

    constraint = EventSequenceConstraint()
    # Use the same subsequence string syntax that TraMineR expects
    contains = check_event_subsequence_containment(
        eseq,
        subseq="(A)-(B)",
        constraint=constraint,
    )

    assert len(contains) == len(eseq)
    # Use equality checks instead of identity to avoid numpy boolean gotchas
    assert contains.iloc[0] is True or bool(contains.iloc[0]) is True
    assert contains.iloc[1] is False or bool(contains.iloc[1]) is False


def _load_additional_ref_if_exists():
    """
    Load additional TraMineR reference CSVs if they exist.

    The R script is expected to write (in this directory):
        - ref_eseq_tse.csv      : TSE table (id, timestamp, event)
        - ref_eseq_etm.csv      : event transition matrix (row/col = events)
        - ref_eseq_contain.csv  : logical vector 'contains_subseq'
    """
    tse_path = os.path.join(TEST_DIR, "ref_eseq_tse.csv")
    etm_path = os.path.join(TEST_DIR, "ref_eseq_etm.csv")
    contain_path = os.path.join(TEST_DIR, "ref_eseq_contain.csv")

    if not (os.path.isfile(tse_path) and os.path.isfile(etm_path) and os.path.isfile(contain_path)):
        return None

    tse = pd.read_csv(tse_path)
    etm = pd.read_csv(etm_path, index_col=0)
    contain = pd.read_csv(contain_path)
    return {"tse": tse, "etm": etm, "contain": contain}


def test_convert_event_sequences_to_tse_traminer_reference(lsog_seqdata):
    """
    When TraMineR reference TSE CSV is present, our
    convert_event_sequences_to_tse() output should match it exactly.

    This test mirrors a typical R pipeline:

        tse <- seqe2tse(eseq)
        write.csv(tse, "ref_eseq_tse.csv", row.names=FALSE)
    """
    from sequenzo.with_event_history_analysis import create_event_sequences

    ref = _load_additional_ref_if_exists()
    if ref is None:
        pytest.skip("Additional TraMineR reference files not found (ref_eseq_tse/etm/contain).")

    # Build the same event sequences as in the R script:
    # LSOG state sequences -> event sequences with tevent='transition'
    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    df = df.head(20)
    states = [1, 2, 3, 4, 5, 6]
    seqdata = SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )
    eseq = create_event_sequences(data=seqdata, tevent="transition")

    tse_out = convert_event_sequences_to_tse(eseq)

    # Sort and align columns exactly with reference before comparison
    ref_tse = ref["tse"].copy()
    tse_sorted = (
        tse_out.sort_values(["id", "timestamp", "event"])
        .reset_index(drop=True)
        .astype({"id": int, "timestamp": float, "event": str})
    )
    ref_sorted = (
        ref_tse.sort_values(["id", "timestamp", "event"])
        .reset_index(drop=True)
        .astype({"id": int, "timestamp": float, "event": str})
    )

    pd.testing.assert_frame_equal(tse_sorted, ref_sorted)


def test_compute_event_transition_matrix_traminer_reference(lsog_seqdata):
    """
    When TraMineR reference ETM CSV is present, our
    compute_event_transition_matrix() output should match it.

    In R, the reference can be generated with, e.g.:

        eseq <- seqecreate(seqdata, tevent=\"transition\")
        etm <- seqetm(eseq)
        write.csv(etm, \"ref_eseq_etm.csv\")
    """
    from sequenzo.with_event_history_analysis import create_event_sequences

    ref = _load_additional_ref_if_exists()
    if ref is None:
        pytest.skip("Additional TraMineR reference files not found (ref_eseq_tse/etm/contain).")

    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    df = df.head(20)
    states = [1, 2, 3, 4, 5, 6]
    seqdata = SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )
    eseq = create_event_sequences(data=seqdata, tevent="transition")

    etm_out = compute_event_transition_matrix(eseq, weighted=True, normalize=True)

    ref_etm = ref["etm"].copy()

    # Align row/column order by event labels (dictionary / colnames in R)
    # We expect that the R script uses the same alphabet / event labels.
    ref_etm = ref_etm.loc[etm_out.index, etm_out.columns]

    pd.testing.assert_frame_equal(
        etm_out.astype(float),
        ref_etm.astype(float),
        check_less_precise=True,
    )


def test_check_event_subsequence_containment_traminer_reference(lsog_seqdata):
    """
    When TraMineR reference containment CSV is present, our
    check_event_subsequence_containment() result should match it.

    In R, one possible way to generate such a reference is:

        eseq <- seqecreate(seqdata, tevent=\"transition\")
        subseq <- \"(A)-(B)\"  # or any valid subsequence string
        contains <- seqecontain(eseq, subseq)
        write.csv(data.frame(contains_subseq=contains),
                  \"ref_eseq_contain.csv\", row.names=FALSE)
    """
    from sequenzo.with_event_history_analysis import create_event_sequences

    ref = _load_additional_ref_if_exists()
    if ref is None:
        pytest.skip("Additional TraMineR reference files not found (ref_eseq_tse/etm/contain).")

    ref_contain = ref["contain"]
    if "contains_subseq" not in ref_contain.columns:
        pytest.skip("ref_eseq_contain.csv must have a 'contains_subseq' column.")

    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    df = df.head(20)
    states = [1, 2, 3, 4, 5, 6]
    seqdata = SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )
    eseq = create_event_sequences(data=seqdata, tevent="transition")

    constraint = EventSequenceConstraint()

    contains = check_event_subsequence_containment(
        eseq,
        subseq="(A)-(B)",  # must match the subsequence used in the R script
        constraint=constraint,
    )

    # Align and compare as boolean vectors
    expected = ref_contain["contains_subseq"].astype(bool).values
    assert len(expected) == len(contains)
    np.testing.assert_array_equal(contains.values.astype(bool), expected)

