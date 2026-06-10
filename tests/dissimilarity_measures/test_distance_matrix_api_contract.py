import importlib
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix


def _seqdata_with_duplicate_reference():
    time_cols = ["T1", "T2", "T3", "T4"]
    raw = pd.DataFrame(
        [
            ["A", "A", "B", "B"],
            ["A", "A", "B", "B"],
            ["B", "B", "A", "A"],
            ["A", "B", "A", "B"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["s0", "s1", "s2", "s3"])
    return SequenceData(raw, time=time_cols, states=["A", "B"], id_col="id")


def _seqdata_all_unique_reference():
    time_cols = ["T1", "T2", "T3", "T4", "T5"]
    raw = pd.DataFrame(
        [
            ["A", "A", "B", "B", "A"],
            ["A", "B", "B", "A", "B"],
            ["B", "A", "A", "B", "B"],
            ["B", "B", "A", "A", "A"],
            ["A", "B", "A", "B", "A"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["u0", "u1", "u2", "u3", "u4"])
    return SequenceData(raw, time=time_cols, states=["A", "B"], id_col="id")


def _seqdata_with_weights_reference():
    time_cols = ["T1", "T2", "T3", "T4", "T5", "T6"]
    raw = pd.DataFrame(
        [
            ["A", "A", "B", "C", "C", "A"],
            ["A", "A", "B", "C", "C", "A"],
            ["B", "C", "C", "A", "B", "B"],
            ["C", "B", "A", "A", "C", "C"],
            ["A", "C", "B", "B", "A", "C"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", [f"w{i}" for i in range(len(raw))])
    return SequenceData(
        raw,
        time=time_cols,
        states=["A", "B", "C"],
        id_col="id",
        weights=[1.0, 2.0, 0.5, 3.0, 1.5],
    )


def _seqdata_ham_asymmetric_reference():
    time_cols = ["T1", "T2"]
    raw = pd.DataFrame(
        [
            ["A", "A"],
            ["B", "A"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["aa", "ba"])
    return SequenceData(raw, time=time_cols, states=["A", "B"], id_col="id")


def _twed_seqdata():
    time_cols = ["T1", "T2", "T3", "T4", "T5"]
    raw = pd.DataFrame(
        [
            [1, 1, 2, 2, 1],
            [1, 2, 2, 1, 1],
            [2, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["s0", "s1", "s2", "s3"])
    return SequenceData(raw, time=time_cols, states=[1, 2], id_col="id")


def _python_lcs_length(left, right):
    prev = [0] * (len(right) + 1)
    curr = [0] * (len(right) + 1)
    for value in left:
        curr[0] = 0
        for j, other in enumerate(right, start=1):
            if value == other:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len(right)]


def _python_lcs_distance_matrix(rows, scale=1.0):
    rows = [list(row) for row in rows]
    out = np.zeros((len(rows), len(rows)), dtype=np.float64)
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            right = rows[j]
            lcs = _python_lcs_length(left, right)
            out[i, j] = scale * (len(left) + len(right) - 2 * lcs)
            out[j, i] = out[i, j]
    return out


def _numeric_seqdata_from_rows(rows, prefix="s"):
    rows = np.asarray(rows, dtype=int)
    time_cols = [f"T{i + 1}" for i in range(rows.shape[1])]
    raw = pd.DataFrame(rows, columns=time_cols)
    raw.insert(0, "id", [f"{prefix}{i}" for i in range(rows.shape[0])])
    states = sorted(np.unique(rows).tolist())
    return raw, time_cols, SequenceData(raw, time=time_cols, states=states, id_col="id")


def test_refseq_int_maps_original_index_to_unique_sequence_with_duplicates():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OM", sm="CONSTANT", indel=1.0, norm="none")

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=0, **kwargs)

    assert isinstance(ref, pd.Series)
    assert list(ref.index) == list(seqdata.ids)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 0].to_numpy(dtype=np.float64))
    assert ref.iloc[0] == 0.0
    assert ref.iloc[1] == 0.0


@pytest.mark.parametrize("refseq", [None, 0, [[1], [0]]])
def test_ham_rejects_asymmetric_substitution_costs(refseq):
    seqdata = _seqdata_ham_asymmetric_reference()
    sm = np.array(
        [
            [0.0, 3.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )

    kwargs = dict(method="HAM", sm=sm, norm="none")
    if refseq is not None:
        kwargs["refseq"] = refseq

    with pytest.raises(ValueError, match="symmetric"):
        get_distance_matrix(seqdata, **kwargs)


def test_omstran_rejects_asymmetric_source_substitution_costs():
    seqdata = _seqdata_ham_asymmetric_reference()
    sm = np.array([[0.0, 3.0], [5.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric"):
        get_distance_matrix(seqdata, method="OMstran", sm=sm, indel=1.0, norm="none")


def test_dhd_accepts_symmetric_time_varying_substitution_array():
    seqdata = _seqdata_with_duplicate_reference()
    base = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    sm = np.repeat(base[None, :, :], 4, axis=0)

    full = get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none")
    condensed = get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none", full_matrix=False)

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)


def test_dhd_time_varying_sm_keeps_equal_states_zero_even_with_nonzero_diagonal():
    time_cols = ["T1", "T2", "T3", "T4", "T5"]
    raw = pd.DataFrame(
        [
            ["A", "A", "B", "B", "A"],
            ["A", "B", "B", "A", "B"],
            ["B", "A", "A", "B", "B"],
            ["B", "B", "A", "A", "A"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["d0", "d1", "d2", "d3"])
    seqdata = SequenceData(raw, time=time_cols, states=["A", "B"], id_col="id")
    sm = np.zeros((len(time_cols), 2, 2), dtype=np.float64)
    for t in range(len(time_cols)):
        sm[t] = np.array(
            [
                [50.0 + t, 2.0 + t],
                [2.0 + t, 80.0 + t],
            ],
            dtype=np.float64,
        )

    labels = raw[time_cols].to_numpy()
    state_idx = {"A": 0, "B": 1}
    expected = np.zeros((len(raw), len(raw)), dtype=np.float64)
    for i in range(len(raw)):
        for j in range(len(raw)):
            total = 0.0
            for t in range(len(time_cols)):
                left = labels[i, t]
                right = labels[j, t]
                if left != right:
                    total += sm[t, state_idx[left], state_idx[right]]
            expected[i, j] = total

    full = get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none")
    condensed = get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none", full_matrix=False)
    ref = get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none", refseq=0)

    np.testing.assert_allclose(full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(condensed, expected[np.triu_indices(len(raw), k=1)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), expected[:, 0], rtol=1e-10, atol=1e-10)


def test_dhd_rejects_asymmetric_time_varying_substitution_array():
    seqdata = _seqdata_with_duplicate_reference()
    sm = np.repeat(np.array([[[0.0, 2.0], [3.0, 0.0]]], dtype=np.float64), 4, axis=0)

    with pytest.raises(ValueError, match="symmetric"):
        get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none")


def test_dhd_rejects_short_time_varying_substitution_array():
    seqdata = _seqdata_with_duplicate_reference()
    base = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    sm = np.repeat(base[None, :, :], 3, axis=0)

    with pytest.raises(ValueError, match="time point"):
        get_distance_matrix(seqdata, method="DHD", sm=sm, norm="none")


def test_refseq_sets_still_returns_requested_block():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OM", sm="CONSTANT", indel=1.0, norm="none")

    full = get_distance_matrix(seqdata, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 1], [2, 3]], **kwargs)

    assert isinstance(block, pd.DataFrame)
    assert block.shape == (2, 2)
    expected = full.iloc[[0, 1], [2, 3]]
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))


@pytest.mark.parametrize("method", ["LCP", "RLCP"])
def test_lcp_refseq_index_uses_reference_path_and_matches_full(monkeypatch, method):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    real_lcp = c_code.LCPdistance
    calls = {"all": 0, "ref": 0}

    class SpyLCPDistance:
        def __init__(self, *args, **kwargs):
            self._inner = real_lcp(*args, **kwargs)

        def compute_all_distances(self):
            calls["all"] += 1
            return self._inner.compute_all_distances()

        def compute_refseq_distances(self):
            calls["ref"] += 1
            return self._inner.compute_refseq_distances()

    full = get_distance_matrix(seqdata, method=method, norm="none")
    monkeypatch.setattr(c_code, "LCPdistance", SpyLCPDistance)

    ref = get_distance_matrix(seqdata, method=method, norm="none", refseq=2)

    assert calls == {"all": 0, "ref": 1}
    assert isinstance(ref, pd.Series)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 2].to_numpy(dtype=np.float64))


@pytest.mark.parametrize("method", ["LCP", "RLCP"])
def test_lcp_refseq_sets_match_full_matrix_block(method):
    seqdata = _seqdata_with_duplicate_reference()

    full = get_distance_matrix(seqdata, method=method, norm="none")
    block = get_distance_matrix(seqdata, method=method, norm="none", refseq=[[0, 1], [2, 3]])

    assert isinstance(block, pd.DataFrame)
    assert block.shape == (2, 2)
    expected = full.iloc[[0, 1], [2, 3]]
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))


def test_omspell_refseq_matches_full_matrix():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OMspell", sm="CONSTANT", indel=1.0, norm="none")

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=2, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 1], [2, 3]], **kwargs)

    assert isinstance(ref, pd.Series)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 2].to_numpy(dtype=np.float64))
    expected = full.iloc[[0, 1], [2, 3]]
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))


def test_svrspell_refseq_and_condensed_match_full_matrix():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="SVRspell", norm="none", tpow=1.0)

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=2, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 1], [2, 3]], **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    assert isinstance(ref, pd.Series)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 2].to_numpy(dtype=np.float64))
    expected = full.iloc[[0, 1], [2, 3]]
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))
    np.testing.assert_allclose(
        condensed,
        full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)],
    )


def test_svrspell_full_matrix_false_uses_condensed_kernel_for_duplicate_mapping(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    captured = {}

    class FakeSVRspellDistance:
        def __init__(self, sequences, seqdur, seqlength, prox, kweights, norm_num, refseq_id):
            captured["n_unique"] = sequences.shape[0]

        def compute_all_distances(self):
            raise AssertionError("SVRspell full_matrix=False must not allocate the unique U x U matrix")

        def compute_condensed_distances(self):
            captured["condensed_called"] = True
            assert captured["n_unique"] == 3
            return np.array([100.0, 400.0, 900.0], dtype=np.float64)

    monkeypatch.setattr(c_code, "SVRspellDistance", FakeSVRspellDistance)

    condensed = get_distance_matrix(
        seqdata,
        method="SVRspell",
        norm="none",
        full_matrix=False,
    )

    assert captured["condensed_called"] is True
    np.testing.assert_allclose(
        condensed,
        np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=np.float64),
    )


def test_unique_condensed_expansion_uses_cpp_helper(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")
    calls = {"count": 0}

    def fake_expand(nseqs, seqdata_didxs, unique_condensed, nunique):
        calls["count"] += 1
        assert nseqs == 4
        np.testing.assert_array_equal(seqdata_didxs, np.array([0, 0, 1, 2], dtype=np.int32))
        np.testing.assert_allclose(unique_condensed, np.array([10.0, 20.0, 30.0]))
        assert nunique == 3
        return np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=np.float64)

    monkeypatch.setattr(c_code, "_unique_condensed_to_condensed", fake_expand, raising=False)

    expanded = gdm_module._unique_condensed_to_condensed(
        4,
        np.array([0, 0, 1, 2], dtype=np.int32),
        np.array([10.0, 20.0, 30.0], dtype=np.float64),
        nunique=3,
    )

    assert calls == {"count": 1}
    np.testing.assert_allclose(expanded, np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0]))


def test_cpp_unique_condensed_expansion_matches_duplicate_order():
    import sequenzo.dissimilarity_measures.c_code as c_code

    expanded = c_code._unique_condensed_to_condensed(
        5,
        np.array([2, 0, 2, 1, 0], dtype=np.int32),
        np.array([10.0, 20.0, 30.0], dtype=np.float64),
        3,
    )

    np.testing.assert_allclose(
        expanded,
        np.array([20.0, 0.0, 30.0, 20.0, 20.0, 10.0, 0.0, 30.0, 20.0, 10.0]),
    )


def test_cpp_unique_sequences_with_indices_preserves_first_occurrence_order():
    import sequenzo.dissimilarity_measures.c_code as c_code

    sequences = np.array(
        [
            [1, 1, 2, 0],
            [2, 2, 1, 0],
            [1, 1, 2, 0],
            [1, 2, 1, 2],
            [2, 2, 1, 0],
        ],
        dtype=np.int32,
    )

    old_unique, old_inverse, old_lengths = c_code.find_unique_sequences(sequences)
    unique, inverse, lengths, first_indices = c_code.find_unique_sequences_with_indices(sequences)

    np.testing.assert_array_equal(unique, old_unique)
    np.testing.assert_array_equal(inverse, old_inverse)
    np.testing.assert_array_equal(lengths, old_lengths)
    np.testing.assert_array_equal(first_indices, np.array([0, 1, 3], dtype=np.int32))
    np.testing.assert_array_equal(unique, sequences[first_indices])


@pytest.mark.parametrize(
    "prox",
    [
        np.eye(1, dtype=np.float64),
        np.ones((2, 3), dtype=np.float64),
        np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64),
    ],
)
def test_svrspell_rejects_invalid_prox_matrix(prox):
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match="SVRspell prox"):
        get_distance_matrix(
            seqdata,
            method="SVRspell",
            norm="none",
            prox=prox,
            full_matrix=False,
        )


@pytest.mark.parametrize("norm", ["none", "auto"])
def test_svrspell_custom_prox_kweights_refseq_matches_full_matrix(norm):
    seqdata = _seqdata_with_duplicate_reference()
    prox = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    kwargs = dict(
        method="SVRspell",
        norm=norm,
        tpow=0.5,
        prox=prox,
        kweights=np.array([2.0, 0.5], dtype=np.float64),
    )

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=3, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 2], [1, 3]], **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 3].to_numpy(dtype=np.float64))
    np.testing.assert_allclose(
        block.to_numpy(dtype=np.float64),
        full.iloc[[0, 2], [1, 3]].to_numpy(dtype=np.float64),
    )
    np.testing.assert_allclose(
        condensed,
        full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)],
    )


def test_ham_uses_dedicated_kernel_and_matches_full_outputs(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    real_ham = c_code.HAMdistance
    calls = {"ham": 0, "dhd": 0}

    class SpyHAMDistance:
        def __init__(self, sequences, sm, norm_num, max_cost, refseq_id):
            calls["ham"] += 1
            assert sm.ndim == 2
            assert np.isfinite(max_cost)
            self._inner = real_ham(sequences, sm, norm_num, max_cost, refseq_id)

        def compute_all_distances(self):
            return self._inner.compute_all_distances()

        def compute_condensed_distances(self):
            return self._inner.compute_condensed_distances()

        def compute_refseq_distances(self):
            return self._inner.compute_refseq_distances()

    class SpyDHDDistance:
        def __init__(self, *args, **kwargs):
            calls["dhd"] += 1
            raise AssertionError("HAM must not route through DHDdistance")

    monkeypatch.setattr(c_code, "HAMdistance", SpyHAMDistance)
    monkeypatch.setattr(c_code, "DHDdistance", SpyDHDDistance)

    sm = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    full = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="auto")
    condensed = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="auto", full_matrix=False)
    ref = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="auto", refseq=2)
    block = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="auto", refseq=[[0, 1], [2, 3]])

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 2].to_numpy(dtype=np.float64))
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), full.iloc[[0, 1], [2, 3]].to_numpy(dtype=np.float64))
    assert calls["ham"] == 4
    assert calls["dhd"] == 0


def test_ham_unpadded_substitution_matrix_matches_state_codes():
    seqdata = _seqdata_with_duplicate_reference()
    sm = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    expected = np.array(
        [
            [0.0, 0.0, 12.0, 6.0],
            [0.0, 0.0, 12.0, 6.0],
            [12.0, 12.0, 0.0, 6.0],
            [6.0, 6.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )

    full = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="none")
    condensed = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="none", full_matrix=False)
    ref = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="none", refseq=2)

    np.testing.assert_allclose(full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(condensed, expected[np.triu_indices(len(seqdata.ids), k=1)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), expected[:, 2], rtol=1e-10, atol=1e-10)


def test_ham_padded_substitution_matrix_matches_unpadded_matrix():
    seqdata = _seqdata_with_duplicate_reference()
    sm = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    padded_sm = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 0.0, 3.0],
            [np.nan, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    full = get_distance_matrix(seqdata, method="HAM", sm=sm, norm="none")
    padded_full = get_distance_matrix(seqdata, method="HAM", sm=padded_sm, norm="none")
    padded_condensed = get_distance_matrix(seqdata, method="HAM", sm=padded_sm, norm="none", full_matrix=False)
    padded_ref = get_distance_matrix(seqdata, method="HAM", sm=padded_sm, norm="none", refseq=2)

    expected = full.to_numpy(dtype=np.float64)
    np.testing.assert_allclose(padded_full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        padded_condensed,
        expected[np.triu_indices(len(seqdata.ids), k=1)],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(padded_ref.to_numpy(dtype=np.float64), expected[:, 2], rtol=1e-10, atol=1e-10)


def test_omtspell_direct_method_accepts_tokdep_coeff():
    seqdata = _seqdata_all_unique_reference()

    direct = get_distance_matrix(
        seqdata,
        method="OMtspell",
        sm="CONSTANT",
        indel=1.0,
        tokdep_coeff=np.ones(2),
        norm="none",
        full_matrix=False,
    )
    switched = get_distance_matrix(
        seqdata,
        method="OMspell",
        sm="CONSTANT",
        indel=1.0,
        tokdep_coeff=np.ones(2),
        norm="none",
        full_matrix=False,
    )

    np.testing.assert_allclose(direct, switched, rtol=1e-10, atol=1e-10)


PAIRWISE_CONDENSED_CASES = [
    ("OM", dict(sm="CONSTANT", indel=1.0, norm="none")),
    ("OMloc", dict(sm="CONSTANT", indel=1.0, norm="none")),
    ("OMslen", dict(sm="CONSTANT", indel=1.0, norm="none")),
    ("OMspell", dict(sm="CONSTANT", indel=1.0, norm="none", tokdep_coeff=np.ones(2))),
    ("HAM", dict(sm="CONSTANT", norm="none")),
    ("DHD", dict(sm="TRATE", norm="none")),
    ("LCS", dict(norm="none")),
    ("LCP", dict(norm="none")),
    ("RLCP", dict(norm="none")),
    ("LCPspell", dict(norm="none")),
    ("RLCPspell", dict(norm="none")),
    ("LCPmst", dict(norm="none")),
    ("LCPprod", dict(norm="none")),
    ("OMspell", dict(sm="CONSTANT", indel=1.0, norm="none")),
    ("OMspellRS", dict(sm="CONSTANT", indel=1.0, norm="none")),
    ("SVRspell", dict(norm="none", tpow=1.0)),
    (
        "TWED",
        dict(
            sm=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64),
            indel=2.0,
            nu=0.5,
            h=0.5,
            norm="none",
        ),
    ),
    ("NMS", dict(norm="none")),
    ("NMSMST", dict(norm="none")),
    ("CHI2", dict(norm="none")),
]


@pytest.mark.parametrize(("method", "kwargs"), PAIRWISE_CONDENSED_CASES)
def test_full_matrix_false_matches_full_upper_triangle_across_methods(method, kwargs):
    seqdata = _seqdata_with_duplicate_reference()

    full = get_distance_matrix(seqdata, method=method, **kwargs)
    condensed = get_distance_matrix(seqdata, method=method, full_matrix=False, **kwargs)

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(("method", "kwargs"), PAIRWISE_CONDENSED_CASES)
def test_full_matrix_false_matches_full_upper_triangle_when_all_sequences_are_unique(method, kwargs):
    seqdata = _seqdata_all_unique_reference()

    full = get_distance_matrix(seqdata, method=method, **kwargs)
    condensed = get_distance_matrix(seqdata, method=method, full_matrix=False, **kwargs)

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(("method", "kwargs"), PAIRWISE_CONDENSED_CASES)
def test_refseq_middle_index_matches_full_column_across_methods(method, kwargs):
    seqdata = _seqdata_all_unique_reference()
    refseq = 2

    full = get_distance_matrix(seqdata, method=method, **kwargs)
    ref = get_distance_matrix(seqdata, method=method, refseq=refseq, **kwargs)

    assert isinstance(ref, pd.Series)
    assert list(ref.index) == list(seqdata.ids)
    np.testing.assert_allclose(
        ref.to_numpy(dtype=np.float64),
        full.iloc[:, refseq].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="NMS", norm="none", prox=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)),
        dict(
            method="SVRspell",
            norm="auto",
            tpow=0.5,
            prox=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64),
            kweights=np.array([2.0, 0.5], dtype=np.float64),
        ),
    ],
)
def test_refseq_middle_index_matches_full_column_for_proximity_methods(kwargs):
    seqdata = _seqdata_all_unique_reference()
    refseq = 2

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=refseq, **kwargs)

    assert isinstance(ref, pd.Series)
    assert list(ref.index) == list(seqdata.ids)
    np.testing.assert_allclose(
        ref.to_numpy(dtype=np.float64),
        full.iloc[:, refseq].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_chi2_full_matrix_false_uses_condensed_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    calls = {"condensed": 0, "all": 0}

    class FakeCHI2Distance:
        def __init__(self, allmat, pdotj, norm_factor, refseq_id):
            self.n = allmat.shape[0]

        def compute_all_distances(self):
            calls["all"] += 1
            raise AssertionError("CHI2 full_matrix=False must not allocate an n x n matrix")

        def compute_condensed_distances(self):
            calls["condensed"] += 1
            return np.arange(self.n * (self.n - 1) // 2, dtype=np.float64)

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    monkeypatch.setattr(c_code, "CHI2distance", FakeCHI2Distance)

    condensed = get_distance_matrix(seqdata, method="CHI2", norm="none", full_matrix=False)

    assert calls == {"condensed": 1, "all": 0}
    np.testing.assert_allclose(condensed, np.arange(6, dtype=np.float64))


def test_lcs_pairwise_uses_lcsdistance_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    calls = {"lcs": 0, "om": 0}

    class FakeLCSDistance:
        def __init__(self, sequences, lengths, norm, refseq_id, distance_scale=1.0):
            calls["lcs"] += 1
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            raise AssertionError("full matrix path is not part of this test")

        def compute_condensed_distances(self):
            return np.arange(self.n * (self.n - 1) // 2, dtype=np.float64)

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    class FakeOMDistance:
        def __init__(self, *args, **kwargs):
            calls["om"] += 1
            raise AssertionError("LCS should not dispatch through OMdistance")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    condensed = get_distance_matrix(seqdata, method="LCS", norm="none", full_matrix=False)

    assert calls == {"lcs": 1, "om": 0}
    np.testing.assert_allclose(condensed, np.array([0.0, 0.0, 1.0, 0.0, 1.0, 2.0]))


def test_lcs_matches_python_reference_for_raw_distance():
    time_cols = ["T1", "T2", "T3", "T4", "T5", "T6"]
    raw = pd.DataFrame(
        [
            [1, 2, 1, 3, 2, 1],
            [1, 1, 3, 2, 2, 1],
            [3, 2, 1, 1, 2, 3],
            [2, 3, 3, 1, 1, 2],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", [f"l{i}" for i in range(len(raw))])
    seqdata = SequenceData(raw, time=time_cols, states=[1, 2, 3], id_col="id")
    expected = _python_lcs_distance_matrix(raw[time_cols].to_numpy())

    full = get_distance_matrix(seqdata, method="LCS", norm="none")
    condensed = get_distance_matrix(seqdata, method="LCS", norm="none", full_matrix=False)
    ref = get_distance_matrix(seqdata, method="LCS", norm="none", refseq=1)

    np.testing.assert_allclose(full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(condensed, expected[np.triu_indices(len(raw), k=1)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), expected[:, 1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_lcs_word_boundary_lengths_match_python_reference(length):
    base = (np.arange(length) % 13) + 1
    rows = np.vstack(
        [
            base,
            np.roll(base, 1),
            base[::-1],
            ((np.arange(length) * 5 + 3) % 13) + 1,
        ]
    )
    raw, time_cols, seqdata = _numeric_seqdata_from_rows(rows, prefix=f"l{length}_")
    expected = _python_lcs_distance_matrix(raw[time_cols].to_numpy())

    full = get_distance_matrix(seqdata, method="LCS", norm="none")
    condensed = get_distance_matrix(seqdata, method="LCS", norm="none", full_matrix=False)
    ref = get_distance_matrix(seqdata, method="LCS", norm="none", refseq=2)
    block = get_distance_matrix(seqdata, method="LCS", norm="none", refseq=[[0, 2], [1, 3]])

    np.testing.assert_allclose(full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(condensed, expected[np.triu_indices(len(raw), k=1)], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), expected[:, 2], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected[np.ix_([0, 2], [1, 3])], rtol=1e-10, atol=1e-10)


def test_full_matrix_false_all_unique_uses_unique_condensed_directly(monkeypatch):
    import importlib
    import sequenzo.dissimilarity_measures.c_code as c_code

    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")
    seqdata = _seqdata_all_unique_reference()
    calls = {"lcs": 0}

    class FakeLCSDistance:
        def __init__(self, sequences, lengths, norm, refseq_id, distance_scale=1.0):
            calls["lcs"] += 1
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            raise AssertionError("full matrix path is not part of this test")

        def compute_condensed_distances(self):
            return np.arange(self.n * (self.n - 1) // 2, dtype=np.float64)

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    def forbidden_expansion(*args, **kwargs):
        raise AssertionError("all-unique condensed output should not be expanded again")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(gdm_module, "_unique_condensed_to_condensed", forbidden_expansion)

    condensed = get_distance_matrix(seqdata, method="LCS", norm="none", full_matrix=False)

    assert calls == {"lcs": 1}
    np.testing.assert_allclose(condensed, np.arange(10, dtype=np.float64))


def test_om_constant_indel_only_case_uses_lcsdistance_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_all_unique_reference()
    calls = {"lcs": 0, "om": 0, "scale": None}

    class FakeLCSDistance:
        def __init__(self, sequences, lengths, norm, refseq_id, distance_scale=1.0):
            calls["lcs"] += 1
            calls["scale"] = distance_scale
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            out = np.zeros((self.n, self.n), dtype=np.float64)
            out[np.triu_indices(self.n, k=1)] = 1.0
            out += out.T
            return out

        def compute_condensed_distances(self):
            raise AssertionError("condensed path is not part of this test")

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    class FakeOMDistance:
        def __init__(self, *args, **kwargs):
            calls["om"] += 1
            raise AssertionError("OM CONSTANT with substitution >= 2 * indel should use the LCS fast path")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    full = get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=1.0, norm="none")

    assert calls == {"lcs": 1, "om": 0, "scale": 1.0}
    assert isinstance(full, pd.DataFrame)
    assert full.shape == (len(seqdata.ids), len(seqdata.ids))


def test_om_lcs_fast_path_scales_nonunit_indel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_all_unique_reference()
    calls = {"lcs": 0, "om": 0, "scale": None}
    sm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )

    class FakeLCSDistance:
        def __init__(self, sequences, lengths, norm, refseq_id, distance_scale=1.0):
            calls["lcs"] += 1
            calls["scale"] = distance_scale
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.n, self.n), dtype=np.float64)

        def compute_condensed_distances(self):
            raise AssertionError("condensed path is not part of this test")

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    class FakeOMDistance:
        def __init__(self, *args, **kwargs):
            calls["om"] += 1
            raise AssertionError("OM with substitution >= 2 * indel should use the LCS fast path")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    full = get_distance_matrix(seqdata, method="OM", sm=sm, indel=2.0, norm="none")

    assert calls == {"lcs": 1, "om": 0, "scale": 2.0}
    assert isinstance(full, pd.DataFrame)


def test_om_user_constant_matrix_matches_constant_cost_results():
    seqdata = _seqdata_all_unique_reference()
    sm = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)

    constant_full = get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=1.0, norm="none")
    matrix_full = get_distance_matrix(seqdata, method="OM", sm=sm, indel=1.0, norm="none")
    matrix_condensed = get_distance_matrix(seqdata, method="OM", sm=sm, indel=1.0, norm="none", full_matrix=False)
    matrix_ref = get_distance_matrix(seqdata, method="OM", sm=sm, indel=1.0, norm="none", refseq=2)

    expected = constant_full.to_numpy(dtype=np.float64)
    np.testing.assert_allclose(matrix_full.to_numpy(dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        matrix_condensed,
        expected[np.triu_indices(len(seqdata.ids), k=1)],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(matrix_ref.to_numpy(dtype=np.float64), expected[:, 2], rtol=1e-10, atol=1e-10)


def test_om_substitution_just_below_two_indels_keeps_general_om_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_all_unique_reference()
    calls = {"lcs": 0, "om": 0}
    sm = np.array([[0.0, 2.0 - 1e-13], [2.0 - 1e-13, 0.0]], dtype=np.float64)

    class FakeLCSDistance:
        def __init__(self, *args, **kwargs):
            calls["lcs"] += 1
            raise AssertionError("substitution below 2 * indel must not use the LCS fast path")

    class FakeOMDistance:
        def __init__(self, sequences, sm, indel, norm, lengths, refseq_id, indellist=None):
            calls["om"] += 1
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.n, self.n), dtype=np.float64)

        def compute_condensed_distances(self):
            raise AssertionError("condensed path is not part of this test")

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    full = get_distance_matrix(seqdata, method="OM", sm=sm, indel=1.0, norm="none")

    assert calls == {"lcs": 0, "om": 1}
    assert isinstance(full, pd.DataFrame)


def test_om_trate_keeps_general_om_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_weights_reference()
    calls = {"lcs": 0, "om": 0}

    class FakeLCSDistance:
        def __init__(self, *args, **kwargs):
            calls["lcs"] += 1
            raise AssertionError("TRATE costs must not use the LCS fast path")

    class FakeOMDistance:
        def __init__(self, sequences, sm, indel, norm, lengths, refseq_id, indellist=None):
            calls["om"] += 1
            assert np.asarray(sm).dtype == np.float64
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.n, self.n), dtype=np.float64)

        def compute_condensed_distances(self):
            raise AssertionError("condensed path is not part of this test")

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    full = get_distance_matrix(seqdata, method="OM", sm="TRATE", indel="auto", norm="none")

    assert calls == {"lcs": 0, "om": 1}
    assert isinstance(full, pd.DataFrame)


def test_om_vector_indel_keeps_general_om_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_all_unique_reference()
    calls = {"lcs": 0, "om": 0}

    class FakeLCSDistance:
        def __init__(self, *args, **kwargs):
            calls["lcs"] += 1
            raise AssertionError("state-specific indel costs must not use the LCS fast path")

    class FakeOMDistance:
        def __init__(self, sequences, sm, indel, norm, lengths, refseq_id, indellist=None):
            calls["om"] += 1
            assert indellist is not None and len(indellist) == 2
            self.n = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.n, self.n), dtype=np.float64)

        def compute_condensed_distances(self):
            raise AssertionError("condensed path is not part of this test")

        def compute_refseq_distances(self):
            raise AssertionError("refseq path is not part of this test")

    monkeypatch.setattr(c_code, "LCSdistance", FakeLCSDistance)
    monkeypatch.setattr(c_code, "OMdistance", FakeOMDistance)

    full = get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=[1.0, 2.0], norm="none")

    assert calls == {"lcs": 0, "om": 1}
    assert isinstance(full, pd.DataFrame)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="CHI2", norm="auto"),
        dict(method="CHI2", norm="auto", step=2),
        dict(method="CHI2", norm="none", step=2, overlap=True),
        dict(method="CHI2", norm="auto", breaks=[(0, 1), (2, 5)]),
        dict(method="CHI2", norm="auto", global_pdotj="obs"),
        dict(method="CHI2", norm="auto", global_pdotj=np.array([0.2, 0.3, 0.5])),
        dict(method="EUCLID", norm="auto", euclid_backend="dense", breaks=[(0, 2), (3, 5)]),
        dict(method="EUCLID", norm="none", euclid_backend="dense", step=2, overlap=True),
    ],
)
def test_chi2_euclid_dense_condensed_matches_full_upper_triangle_for_weighted_windows(kwargs):
    seqdata = _seqdata_with_weights_reference()

    full = get_distance_matrix(seqdata, **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["CHI2", "EUCLID"])
def test_chi2_euclid_refseq_outputs_match_full_matrix(method):
    seqdata = _seqdata_with_weights_reference()
    kwargs = dict(method=method, norm="auto")
    if method == "EUCLID":
        kwargs["euclid_backend"] = "dense"

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=3, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 1], [3, 4]], **kwargs)

    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 3].to_numpy(dtype=np.float64))
    np.testing.assert_allclose(
        block.to_numpy(dtype=np.float64),
        full.iloc[[0, 1], [3, 4]].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_chi2_python_fallback_accepts_global_pdotj_array(monkeypatch):
    import sequenzo.dissimilarity_measures as dissimilarity_pkg

    monkeypatch.setattr(dissimilarity_pkg, "_import_c_code", lambda: None)
    seqdata = _seqdata_with_weights_reference()
    kwargs = dict(method="CHI2", norm="auto", global_pdotj=np.array([0.2, 0.3, 0.5]))

    full = get_distance_matrix(seqdata, **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    np.testing.assert_allclose(condensed, expected, rtol=1e-10, atol=1e-10)


def test_refseq_sets_out_of_range_is_rejected_before_indexing():
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match="out of range"):
        get_distance_matrix(seqdata, refseq=[[0], [4]], method="OM", sm="CONSTANT", indel=1.0, norm="none")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="OM", sm="CONSTANT", indel=1.0, norm="none"),
        dict(method="CHI2", norm="auto"),
        dict(method="EUCLID", norm="auto", euclid_backend="categorical"),
    ],
)
@pytest.mark.parametrize("refseq", [2, [[0, 1], [2, 3]]])
def test_refseq_result_is_independent_of_full_matrix_false(kwargs, refseq):
    seqdata = _seqdata_with_duplicate_reference()

    expected = get_distance_matrix(seqdata, refseq=refseq, **kwargs)
    actual = get_distance_matrix(seqdata, refseq=refseq, full_matrix=False, **kwargs)

    if isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series)
        np.testing.assert_allclose(
            actual.to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )
    else:
        assert isinstance(actual, pd.DataFrame)
        np.testing.assert_allclose(
            actual.to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )


def test_full_matrix_false_fallback_uses_cxx_condensed_padding_without_full_padding(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()

    real_dist2matrix = c_code.dist2matrix
    calls = {"condensed": 0, "matrix": 0}

    class FakeNMSDistance:
        def __init__(self, *args, **kwargs):
            pass

        def compute_all_distances(self):
            return np.array(
                [
                    [0.0, 100.0, 400.0],
                    [100.0, 0.0, 900.0],
                    [400.0, 900.0, 0.0],
                ],
                dtype=np.float64,
            )

    class SpyDist2Matrix:
        def __init__(self, *args, **kwargs):
            self._inner = real_dist2matrix(*args, **kwargs)

        def padding_matrix(self):
            calls["matrix"] += 1
            raise AssertionError("full_matrix=False must not expand to a full matrix")

        def padding_condensed(self):
            calls["condensed"] += 1
            return self._inner.padding_condensed()

    monkeypatch.setattr(c_code, "dist2matrix", SpyDist2Matrix)
    monkeypatch.setattr(c_code, "NMSdistance", FakeNMSDistance)

    condensed = get_distance_matrix(seqdata, method="NMS", norm="none", full_matrix=False)
    expected = np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=np.float64)
    assert calls == {"condensed": 1, "matrix": 0}
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(condensed, expected)


def test_nms_full_matrix_false_uses_direct_condensed_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    real_nms = c_code.NMSdistance
    calls = {"condensed": 0, "all": 0}

    class SpyNMSDistance:
        def __init__(self, *args, **kwargs):
            self._inner = real_nms(*args, **kwargs)

        def compute_all_distances(self):
            calls["all"] += 1
            return self._inner.compute_all_distances()

        def compute_condensed_distances(self):
            calls["condensed"] += 1
            return self._inner.compute_condensed_distances()

        def compute_refseq_distances(self):
            return self._inner.compute_refseq_distances()

    monkeypatch.setattr(c_code, "NMSdistance", SpyNMSDistance)

    condensed = get_distance_matrix(seqdata, method="NMS", norm="none", full_matrix=False)

    assert calls == {"condensed": 1, "all": 0}
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)


def test_nms_with_prox_full_matrix_false_uses_direct_condensed_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    real_soft_nms = c_code.NMSMSTSoftdistanceII
    calls = {"condensed": 0, "all": 0}

    class SpySoftNMSDistance:
        def __init__(self, *args, **kwargs):
            self._inner = real_soft_nms(*args, **kwargs)

        def compute_all_distances(self):
            calls["all"] += 1
            return self._inner.compute_all_distances()

        def compute_condensed_distances(self):
            calls["condensed"] += 1
            return self._inner.compute_condensed_distances()

        def compute_refseq_distances(self):
            return self._inner.compute_refseq_distances()

    monkeypatch.setattr(c_code, "NMSMSTSoftdistanceII", SpySoftNMSDistance)

    full = get_distance_matrix(
        seqdata,
        method="NMS",
        norm="none",
        prox=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64),
    )
    condensed = get_distance_matrix(
        seqdata,
        method="NMS",
        norm="none",
        prox=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64),
        full_matrix=False,
    )

    assert calls == {"condensed": 1, "all": 1}
    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    np.testing.assert_allclose(condensed, expected)


@pytest.mark.parametrize(
    ("prox", "message"),
    [
        (np.ones((3, 3), dtype=np.float64), "shape"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]], dtype=np.float64), "finite"),
        (np.array([[1.0, 0.25], [0.5, 1.0]], dtype=np.float64), "symmetric"),
    ],
)
def test_nms_rejects_invalid_proximity_matrix(prox, message):
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match=message):
        get_distance_matrix(seqdata, method="NMS", norm="none", prox=prox)


def test_svrspell_rejects_asymmetric_proximity_matrix():
    seqdata = _seqdata_with_duplicate_reference()
    prox = np.array([[1.0, 0.25], [0.5, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric"):
        get_distance_matrix(seqdata, method="SVRspell", norm="none", prox=prox)


def test_nmsmst_full_matrix_false_uses_direct_condensed_kernel(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    real_nmsmst = c_code.NMSMSTdistance
    calls = {"condensed": 0, "all": 0}

    class SpyNMSMSTDistance:
        def __init__(self, *args, **kwargs):
            self._inner = real_nmsmst(*args, **kwargs)

        def compute_all_distances(self):
            calls["all"] += 1
            return self._inner.compute_all_distances()

        def compute_condensed_distances(self):
            calls["condensed"] += 1
            return self._inner.compute_condensed_distances()

        def compute_refseq_distances(self):
            return self._inner.compute_refseq_distances()

    monkeypatch.setattr(c_code, "NMSMSTdistance", SpyNMSMSTDistance)

    condensed = get_distance_matrix(seqdata, method="NMSMST", norm="none", full_matrix=False)

    assert calls == {"condensed": 1, "all": 0}
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)


def test_numeric_list_indel_is_accepted():
    seqdata = _seqdata_with_duplicate_reference()
    dist = get_distance_matrix(
        seqdata,
        method="OM",
        sm="CONSTANT",
        indel=[1.0, 2.0],
        norm="none",
    )
    assert dist.shape == (4, 4)


@pytest.mark.parametrize("indel", [np.nan, np.inf, -1.0])
def test_om_rejects_invalid_scalar_indel(indel):
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match="indel"):
        get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=indel, norm="none")


@pytest.mark.parametrize("indel", [[np.nan, 1.0], [np.inf, 1.0], [1.0, -1.0]])
def test_om_rejects_invalid_vector_indel(indel):
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match="indel"):
        get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=indel, norm="none")


def test_twed_matrix_indel_auto_uses_traminer_compatible_formula(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _twed_seqdata()
    sm = np.array([[0.0, 2.0], [2.0, 0.0]])
    nu = 0.5
    h = 0.5
    traminer_compatible_indel = 2 * np.nanmax(sm) + nu + h
    captured = {}

    class FakeTWEDDistance:
        def __init__(self, sequences, sm, indel, norm_num, nu, h_twed, lengths, refseq_id):
            captured["indel"] = indel
            self.nseqs = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.nseqs, self.nseqs), dtype=np.float64)

    monkeypatch.setattr(c_code, "TWEDdistance", FakeTWEDDistance)

    get_distance_matrix(
        seqdata,
        method="TWED",
        sm=sm,
        indel="auto",
        nu=nu,
        h=h,
        norm="none",
    )

    assert captured["indel"] == traminer_compatible_indel


def test_twed_full_matrix_false_uses_condensed_kernel_for_duplicate_mapping(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    captured = {}

    class FakeTWEDDistance:
        def __init__(self, sequences, sm, indel, norm_num, nu, h_twed, lengths, refseq_id):
            captured["n_unique"] = sequences.shape[0]

        def compute_all_distances(self):
            raise AssertionError("TWED full_matrix=False must not allocate the unique U x U matrix")

        def compute_condensed_distances(self):
            captured["condensed_called"] = True
            assert captured["n_unique"] == 3
            # unique pairs: (0,1), (0,2), (1,2)
            return np.array([10.0, 20.0, 30.0], dtype=np.float64)

    monkeypatch.setattr(c_code, "TWEDdistance", FakeTWEDDistance)

    condensed = get_distance_matrix(
        seqdata,
        method="TWED",
        sm=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64),
        indel=2.0,
        nu=0.5,
        h=0.5,
        norm="none",
        full_matrix=False,
    )

    assert captured["condensed_called"] is True
    np.testing.assert_allclose(
        condensed,
        np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=np.float64),
    )


def test_twed_rejects_nonfinite_state_substitution_costs():
    seqdata = _twed_seqdata()
    sm = np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite state-to-state costs"):
        get_distance_matrix(
            seqdata,
            method="TWED",
            sm=sm,
            indel=2.0,
            nu=0.5,
            h=0.5,
            norm="none",
        )


def test_twed_rejects_negative_state_substitution_costs():
    seqdata = _twed_seqdata()
    sm = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="non-negative state-to-state costs"):
        get_distance_matrix(
            seqdata,
            method="TWED",
            sm=sm,
            indel=2.0,
            nu=0.5,
            h=0.5,
            norm="none",
        )


def test_get_distance_matrix_does_not_force_gc_by_default(monkeypatch):
    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")

    seqdata = _seqdata_with_duplicate_reference()
    calls = []

    monkeypatch.setattr(gdm_module.gc, "collect", lambda: calls.append(True) or 0)

    get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=1.0, norm="none")

    assert calls == []


def test_get_distance_matrix_can_opt_in_to_gc(monkeypatch):
    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")

    seqdata = _seqdata_with_duplicate_reference()
    calls = []

    monkeypatch.setattr(gdm_module.gc, "collect", lambda: calls.append(True) or 0)

    get_distance_matrix(
        seqdata,
        method="OM",
        sm="CONSTANT",
        indel=1.0,
        norm="none",
        collect_garbage=True,
    )

    assert calls == [True]


def test_dissimilarity_extension_builds_without_fast_math_for_twed_ieee_semantics():
    setup_py = Path(__file__).resolve().parents[2] / "setup.py"
    source = setup_py.read_text()

    assert "def get_compile_args_for_file(filename, *, fast_math=True):" in source
    assert 'dissimilarity_compile_args = get_compile_args_for_file("dummy.cpp", fast_math=False)' in source
    assert (
        "sources=['sequenzo/dissimilarity_measures/src/module.cpp'],\n"
        "            include_dirs=get_dissimilarity_measures_include_dirs(),\n"
        "            extra_compile_args=dissimilarity_compile_args,"
    ) in source


def test_loaded_dissimilarity_extension_exposes_runtime_metadata():
    import sequenzo.dissimilarity_measures.c_code as c_code

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    module_path = Path(c_code.__file__)
    info = c_code._openmp_runtime_info(1)

    assert module_path.exists()
    assert suffix is None or module_path.name.endswith(suffix)
    assert isinstance(info, dict)
    assert "_OPENMP" in info
    assert "compiler" in info
