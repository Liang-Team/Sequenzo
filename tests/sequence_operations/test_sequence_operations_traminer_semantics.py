import numpy as np
import pandas as pd

from sequenzo import SequenceData
from sequenzo.sequence_operations import (
    concatenate_sequences,
    convert_sequences_to_numeric_matrix,
    decompose_concatenated_sequences,
    find_sequence_occurrences,
    longest_common_prefix_length,
    longest_common_subsequence_length,
    pairwise_sequence_alignment,
    recode_sequence_states,
    shift_sequence_with_missing_padding,
    split_fixed_width_sequences,
)
from sequenzo.dissimilarity_measures import get_substitution_cost_matrix


def test_seqconc_matches_traminer_row_concatenation():
    # TraMineR seqconc: concatenate row values with separator, skipping NA by default.
    df = pd.DataFrame(
        {
            "T1": ["A", "B"],
            "T2": ["B", np.nan],
            "T3": ["C", "A"],
        },
        index=["i1", "i2"],
    )
    out = concatenate_sequences(df, sep="-", vname="Sequence")
    assert list(out.columns) == ["Sequence"]
    assert out.loc["i1", "Sequence"] == "A-B-C"
    assert out.loc["i2", "Sequence"] == "B-A"


def test_seqdecomp_matches_traminer_split_and_pad():
    # TraMineR seqdecomp: split by sep, convert miss token to NA, pad to max length.
    df = pd.DataFrame({"Sequence": ["A-B-C", "B-NA"]}, index=["r1", "r2"])
    out = decompose_concatenated_sequences(df, sep="-", miss="NA")
    assert out.shape == (2, 3)
    assert out.iloc[0].tolist() == ["A", "B", "C"]
    assert out.iloc[1, 0] == "B"
    assert pd.isna(out.iloc[1, 1])
    assert pd.isna(out.iloc[1, 2])


def test_seqsep_matches_traminer_fixed_width_split():
    # TraMineR seqsep: split string every sl chars and join with sep.
    out = split_fixed_width_sequences(["ABCDEF", "ZZ"], sl=2, sep="-")
    assert out == ["AB-CD-EF", "ZZ"]


def test_seqshift_matches_traminer_padding_rule():
    # TraMineR seqshift(seq, nbshift): c(rep(NA, seql-nbshift), seq[1:nbshift]).
    out = shift_sequence_with_missing_padding(["A", "B", "C", "D"], 2)
    assert len(out) == 4
    assert pd.isna(out[0]) and pd.isna(out[1])
    assert out[2:] == ["A", "B"]


def test_seqrecode_dataframe_matches_mapping_semantics():
    # TraMineR seqrecode semantics: recode mapped states, keep others if otherwise is NULL.
    df = pd.DataFrame({"T1": ["A", "C"], "T2": ["B", "A"]})
    recodes = {"X": ["A", "B"]}
    out = recode_sequence_states(df, recodes=recodes, otherwise=None)
    assert out.iloc[0].tolist() == ["X", "X"]
    assert out.iloc[1].tolist() == ["C", "X"]


def test_seqasnum_matches_traminer_zero_based_for_non_missing():
    # TraMineR seqasnum: integer matrix, coding starts at 0 by alphabet order.
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "T1": [1, 2],
            "T2": [2, 1],
            "T3": [1, 2],
        }
    )
    seqdata = SequenceData(data=raw, time=["T1", "T2", "T3"], states=[1, 2], id_col="id")
    out = convert_sequences_to_numeric_matrix(seqdata, with_missing=False)
    # states=[1,2] -> codes [0,1]
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(out, expected)


def test_longest_common_prefix_length_matches_traminer_prefix_length():
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "T1": ["A", "A"],
            "T2": ["B", "B"],
            "T3": ["C", "D"],
        }
    )
    seqdata = SequenceData(data=raw, time=["T1", "T2", "T3"], states=["A", "B", "C", "D"], id_col="id")
    assert longest_common_prefix_length(seqdata, seqdata, 0, 1) == 2


def test_longest_common_subsequence_length_matches_traminer_lcs_length():
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "T1": ["A", "A"],
            "T2": ["B", "C"],
            "T3": ["C", "B"],
            "T4": ["D", "D"],
        }
    )
    seqdata = SequenceData(data=raw, time=["T1", "T2", "T3", "T4"], states=["A", "B", "C", "D"], id_col="id")
    # LCS between A-B-C-D and A-C-B-D is A-C-D (or A-B-D), length 3.
    assert longest_common_subsequence_length(seqdata, seqdata, 0, 1) == 3


def test_find_sequence_occurrences_matches_traminer_occurrence_positions():
    xraw = pd.DataFrame({"id": [1], "T1": ["A"], "T2": ["B"], "T3": ["C"]})
    yraw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "T1": ["A", "A", "B"],
            "T2": ["B", "B", "C"],
            "T3": ["C", "D", "D"],
        }
    )
    x = SequenceData(data=xraw, time=["T1", "T2", "T3"], states=["A", "B", "C", "D"], id_col="id")
    y = SequenceData(data=yraw, time=["T1", "T2", "T3"], states=["A", "B", "C", "D"], id_col="id")
    # Only first sequence in y matches, return 1-based position.
    assert find_sequence_occurrences(x, y) == [1]


def test_pairwise_sequence_alignment_matches_traminer_edit_cost_and_ops():
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "T1": ["A", "A"],
            "T2": ["B", "C"],
            "T3": ["C", "C"],
        }
    )
    seqdata = SequenceData(data=raw, time=["T1", "T2", "T3"], states=["A", "B", "C"], id_col="id")
    sm = get_substitution_cost_matrix(seqdata, method="CONSTANT", cval=2.0)["sm"]
    out = pairwise_sequence_alignment(seqdata, indices=[0, 1], indel=1.0, sm=sm)
    # TraMineR tie-breaking uses substitution first when costs are equal.
    assert out.operation == ["E", "S", "E"]
    assert sum(out.cost) == 2.0
