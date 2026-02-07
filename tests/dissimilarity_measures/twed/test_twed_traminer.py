"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_twed_traminer.py
@Time    : 2026/02/07 20:40
@Desc    : 
TWED vs TraMineR reference: same data and parameters must give identical distances.

Reference values from TraMineR (tests/traminer_twed_reference.R):
  norm="none", nu=0.5, h=0.5, sm=[[0,2],[2,0]], indel=2
  4 sequences: (1,1,2,2,1), (1,2,2,1,1), (2,1,1,2,2), (1,1,1,2,2)
  D[0,1]=5, D[0,2]=9, D[0,3]=5, D[1,2]=13, D[1,3]=9, D[2,3]=4
"""
import numpy as np
import pandas as pd
from sequenzo import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix

# TraMineR reference (upper triangle, 0-based indices)
TRAMINER_TWED_REF = {
    (0, 1): 5.0,
    (0, 2): 9.0,
    (0, 3): 5.0,
    (1, 2): 13.0,
    (1, 3): 9.0,
    (2, 3): 4.0,
}


def _make_twed_test_data():
    raw = np.array(
        [[1, 1, 2, 2, 1], [1, 2, 2, 1, 1], [2, 1, 1, 2, 2], [1, 1, 1, 2, 2]],
        dtype=int,
    )
    time_cols = ["C1", "C2", "C3", "C4", "C5"]
    df = pd.DataFrame(raw, columns=time_cols)
    df.insert(0, "id", ["s0", "s1", "s2", "s3"])
    return SequenceData(df, time=time_cols, states=[1, 2], id_col="id")


def test_twed_matches_traminer_pairwise():
    """Sequenzo TWED (norm=none, nu=0.5, h=0.5, sm constant 2, indel=2) must match TraMineR."""
    seqdata = _make_twed_test_data()
    # 2x2 sm for states 1,2; get_distance_matrix adds dummy row/col for TWED. indel=2 to match R.
    sm = np.array([[0.0, 2.0], [2.0, 0.0]])
    D = get_distance_matrix(
        seqdata,
        method="TWED",
        norm="none",
        nu=0.5,
        h=0.5,
        sm=sm,
        indel=2,
    )
    assert D.shape == (4, 4)
    for (i, j), ref in TRAMINER_TWED_REF.items():
        got = D.iloc[i, j]
        assert np.isclose(got, ref, rtol=0, atol=1e-9), f"D[{i},{j}] got {got}, ref {ref}"


def test_twed_matches_traminer_refseq_sets():
    """TWED with refseq (two sets) should match pairwise block from full matrix."""
    seqdata = _make_twed_test_data()
    sm = np.array([[0.0, 2.0], [2.0, 0.0]])
    kw = dict(method="TWED", norm="none", nu=0.5, h=0.5, sm=sm, indel=2)
    # Full pairwise
    D_full = get_distance_matrix(seqdata, **kw)
    # Two sets: set1 = [0,1], set2 = [2,3]
    D_ref = get_distance_matrix(seqdata, refseq=[[0, 1], [2, 3]], **kw)
    assert D_ref.shape == (2, 2)
    assert np.isclose(D_ref.iloc[0, 0], D_full.iloc[0, 2], rtol=0, atol=1e-9)
    assert np.isclose(D_ref.iloc[0, 1], D_full.iloc[0, 3], rtol=0, atol=1e-9)
    assert np.isclose(D_ref.iloc[1, 0], D_full.iloc[1, 2], rtol=0, atol=1e-9)
    assert np.isclose(D_ref.iloc[1, 1], D_full.iloc[1, 3], rtol=0, atol=1e-9)
