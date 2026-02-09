"""
Tests for event sequence analysis using LSOG (dyadic_children) dataset.

Validates results against TraMineR when reference files are present.
Generate reference with:
  Rscript tests/event_sequence_analysis/traminer_reference_event_sequence.R \\
    sequenzo/datasets/dyadic_children.csv 20 tests/event_sequence_analysis
"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sequenzo import SequenceData
from sequenzo.datasets import load_dataset
from sequenzo.with_event_history_analysis import (
    create_event_sequences,
    find_frequent_subsequences,
    count_subsequence_occurrences,
    compare_groups,
    plot_event_sequences,
    plot_subsequence_frequencies,
    EventSequenceConstraint,
    EventSequence,
    SubsequenceList,
)


# Directory containing this test file (for ref files)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def lsog_seqdata():
    """Load dyadic_children (LSOG) and build SequenceData (same as sequence_characteristics tests)."""
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
def lsog_eseq(lsog_seqdata):
    """Convert LSOG state sequence to event sequence (transition method)."""
    return create_event_sequences(data=lsog_seqdata, tevent="transition")


# =============================================================================
# Core pipeline tests
# =============================================================================


def test_create_event_sequences_from_state(lsog_seqdata):
    """create_event_sequences from state sequence produces EventSequenceList."""
    eseq = create_event_sequences(data=lsog_seqdata, tevent="transition")
    assert eseq is not None
    assert len(eseq) == lsog_seqdata.seqdata.shape[0]
    assert len(eseq.dictionary) > 0
    # Dictionary should contain transition-like labels (e.g. "1>2")
    for name in eseq.dictionary[:3]:
        assert ">" in name or name.isdigit()


def test_find_frequent_subsequences(lsog_eseq):
    """find_frequent_subsequences returns SubsequenceList with Support and Count."""
    fsub = find_frequent_subsequences(lsog_eseq, min_support=2)
    assert fsub is not None
    assert hasattr(fsub, "data")
    assert hasattr(fsub, "subsequences")
    if len(fsub) > 0:
        assert "Support" in fsub.data.columns
        assert "Count" in fsub.data.columns
        assert (fsub.data["Support"] >= 0).all() and (fsub.data["Support"] <= 1).all()
        assert (fsub.data["Count"] >= 0).all()


def test_count_subsequence_occurrences(lsog_eseq):
    """count_subsequence_occurrences returns matrix of correct shape."""
    fsub = find_frequent_subsequences(lsog_eseq, min_support=2)
    pres = count_subsequence_occurrences(fsub, method="presence")
    assert pres.shape[0] == len(lsog_eseq)
    assert pres.shape[1] == len(fsub)
    assert np.all((pres == 0) | (pres == 1))


def test_compare_groups(lsog_eseq):
    """compare_groups returns SubsequenceList with p.value and statistic."""
    fsub = find_frequent_subsequences(lsog_eseq, min_support=2)
    if len(fsub) == 0:
        pytest.skip("No frequent subsequences to compare")
    # Use sex as group (from dyadic_children: column 'sex')
    df = load_dataset("dyadic_children").head(20)
    group = df["sex"].values  # 0/1
    discr = compare_groups(fsub, group, method="chisq", pvalue_limit=1.0)
    assert discr is not None
    if len(discr) > 0:
        assert "p.value" in discr.data.columns
        assert "statistic" in discr.data.columns


# =============================================================================
# TraMineR reference comparison (when ref files exist)
# =============================================================================


def _load_ref_if_exists():
    """Load TraMineR reference files if present."""
    ref_support = os.path.join(TEST_DIR, "ref_eseq_fsub_support.csv")
    ref_apply = os.path.join(TEST_DIR, "ref_eseq_applysub.csv")
    ref_meta = os.path.join(TEST_DIR, "ref_eseq_meta.csv")
    ref_alphabet = os.path.join(TEST_DIR, "ref_eseq_alphabet.csv")
    if not os.path.isfile(ref_support) or not os.path.isfile(ref_apply) or not os.path.isfile(ref_meta):
        return None
    support = pd.read_csv(ref_support)
    applysub = pd.read_csv(ref_apply, index_col=0)
    meta = pd.read_csv(ref_meta)
    out = {"support": support, "applysub": applysub, "meta": meta}
    if os.path.isfile(ref_alphabet):
        out["alphabet"] = pd.read_csv(ref_alphabet)["event"].astype(str).tolist()
    else:
        out["alphabet"] = None
    return out


def _parse_ref_subseq(subseq_str: str, ref_alphabet: list) -> list:
    """Parse TraMineR Subseq string into list of event labels. (1) = ref_alphabet[0], (1>2) = label '1>2'."""
    parts = subseq_str.split(")-(")
    labels = []
    for p in parts:
        p = p.strip("()")
        if p.isdigit():
            labels.append(ref_alphabet[int(p) - 1])
        else:
            labels.append(p)
    return labels


def test_event_sequences_match_traminer_meta(lsog_eseq):
    """Number of sequences and event alphabet size match TraMineR ref."""
    ref = _load_ref_if_exists()
    if ref is None:
        pytest.skip("TraMineR reference not found. Run traminer_reference_event_sequence.R")
    assert len(lsog_eseq) == ref["meta"]["n_sequences"].iloc[0]
    assert len(lsog_eseq.dictionary) == ref["meta"]["n_events"].iloc[0]


def test_event_sequences_match_traminer_fsub(lsog_seqdata):
    """For each TraMineR frequent subsequence, our presence Support and Count match."""
    ref = _load_ref_if_exists()
    if ref is None:
        pytest.skip("TraMineR reference not found. Run traminer_reference_event_sequence.R")
    if ref.get("alphabet") is None:
        pytest.skip("ref_eseq_alphabet.csv not found. Re-run R script to generate it.")

    ref_alphabet = ref["alphabet"]
    eseq = create_event_sequences(data=lsog_seqdata, tevent="transition", alphabet=ref_alphabet)
    ref_sup = ref["support"]
    constraint = EventSequenceConstraint()

    # Build SubsequenceList from ref Subseq strings (same order as TraMineR)
    subsequences = []
    for subseq_str in ref_sup["Subseq"]:
        labels = _parse_ref_subseq(subseq_str, ref_alphabet)
        codes = np.array([eseq.dictionary.index(l) + 1 for l in labels])
        ts = np.arange(1.0, len(labels) + 1.0, dtype=np.float64)
        subsequences.append(EventSequence(-1, ts, codes, eseq.dictionary))

    ref_data = ref_sup[["Support", "Count"]].copy()
    subseq_list = SubsequenceList(eseq, subsequences, ref_data, constraint, "frequent")
    pres = count_subsequence_occurrences(subseq_list, method="presence")

    total_weight = np.sum(eseq.weights)
    our_support = np.sum(eseq.weights[:, np.newaxis] * pres, axis=0) / total_weight
    our_count = np.sum(pres, axis=0)

    np.testing.assert_allclose(our_support, ref_sup["Support"].values, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(our_count, ref_sup["Count"].values, rtol=1e-5, atol=1e-8)


def test_event_sequences_match_traminer_applysub(lsog_seqdata):
    """Presence matrix (rows=sequences, cols=ref subsequences) matches TraMineR."""
    ref = _load_ref_if_exists()
    if ref is None:
        pytest.skip("TraMineR reference not found. Run traminer_reference_event_sequence.R")
    if ref.get("alphabet") is None:
        pytest.skip("ref_eseq_alphabet.csv not found. Re-run R script to generate it.")

    ref_alphabet = ref["alphabet"]
    eseq = create_event_sequences(data=lsog_seqdata, tevent="transition", alphabet=ref_alphabet)
    ref_sup = ref["support"]
    ref_mat = ref["applysub"]
    constraint = EventSequenceConstraint()

    subsequences = []
    for subseq_str in ref_sup["Subseq"]:
        labels = _parse_ref_subseq(subseq_str, ref_alphabet)
        codes = np.array([eseq.dictionary.index(l) + 1 for l in labels])
        ts = np.arange(1.0, len(labels) + 1.0, dtype=np.float64)
        subsequences.append(EventSequence(-1, ts, codes, eseq.dictionary))

    ref_data = ref_sup[["Support", "Count"]].copy()
    subseq_list = SubsequenceList(eseq, subsequences, ref_data, constraint, "frequent")
    pres = count_subsequence_occurrences(subseq_list, method="presence")

    assert pres.shape == ref_mat.shape, (
        f"applysub shape: Sequenzo={pres.shape}, TraMineR={ref_mat.shape}"
    )
    np.testing.assert_allclose(pres, ref_mat.values.astype(np.float64), rtol=1e-5, atol=1e-8)


# =============================================================================
# Visualization tests
# =============================================================================


def test_plot_event_sequences_index(lsog_eseq):
    """plot_event_sequences(type='index') runs and returns a figure."""
    fig = plot_event_sequences(lsog_eseq, type="index", top_n=10)
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_event_sequences_parallel(lsog_eseq):
    """plot_event_sequences(type='parallel') runs and returns a figure."""
    fig = plot_event_sequences(lsog_eseq, type="parallel", top_n=10)
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_subsequence_frequencies(lsog_eseq):
    """plot_subsequence_frequencies runs and returns a figure."""
    fsub = find_frequent_subsequences(lsog_eseq, min_support=2)
    if len(fsub) == 0:
        pytest.skip("No frequent subsequences to plot")
    fig = plot_subsequence_frequencies(fsub, top_n=min(5, len(fsub)))
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_subsequence_frequencies_empty_raises(lsog_eseq):
    """plot_subsequence_frequencies on empty SubsequenceList raises."""
    from sequenzo.with_event_history_analysis import (
        SubsequenceList,
        EventSequenceConstraint,
    )

    empty = SubsequenceList(
        lsog_eseq,
        [],
        pd.DataFrame(),
        EventSequenceConstraint(),
        "frequent",
    )
    with pytest.raises(ValueError, match="empty"):
        plot_subsequence_frequencies(empty)
