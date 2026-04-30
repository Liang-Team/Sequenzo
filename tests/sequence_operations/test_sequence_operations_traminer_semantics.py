import numpy as np
import pandas as pd

from sequenzo import SequenceData
from sequenzo.sequence_operations import (
    seqasnum,
    seqconc,
    seqdecomp,
    seqrecode,
    seqsep,
    seqshift,
)


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
    out = seqconc(df, sep="-", vname="Sequence")
    assert list(out.columns) == ["Sequence"]
    assert out.loc["i1", "Sequence"] == "A-B-C"
    assert out.loc["i2", "Sequence"] == "B-A"


def test_seqdecomp_matches_traminer_split_and_pad():
    # TraMineR seqdecomp: split by sep, convert miss token to NA, pad to max length.
    df = pd.DataFrame({"Sequence": ["A-B-C", "B-NA"]}, index=["r1", "r2"])
    out = seqdecomp(df, sep="-", miss="NA")
    assert out.shape == (2, 3)
    assert out.iloc[0].tolist() == ["A", "B", "C"]
    assert out.iloc[1, 0] == "B"
    assert pd.isna(out.iloc[1, 1])
    assert pd.isna(out.iloc[1, 2])


def test_seqsep_matches_traminer_fixed_width_split():
    # TraMineR seqsep: split string every sl chars and join with sep.
    out = seqsep(["ABCDEF", "ZZ"], sl=2, sep="-")
    assert out == ["AB-CD-EF", "ZZ"]


def test_seqshift_matches_traminer_padding_rule():
    # TraMineR seqshift(seq, nbshift): c(rep(NA, seql-nbshift), seq[1:nbshift]).
    out = seqshift(["A", "B", "C", "D"], 2)
    assert len(out) == 4
    assert pd.isna(out[0]) and pd.isna(out[1])
    assert out[2:] == ["A", "B"]


def test_seqrecode_dataframe_matches_mapping_semantics():
    # TraMineR seqrecode semantics: recode mapped states, keep others if otherwise is NULL.
    df = pd.DataFrame({"T1": ["A", "C"], "T2": ["B", "A"]})
    recodes = {"X": ["A", "B"]}
    out = seqrecode(df, recodes=recodes, otherwise=None)
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
    out = seqasnum(seqdata, with_missing=False)
    # states=[1,2] -> codes [0,1]
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(out, expected)
