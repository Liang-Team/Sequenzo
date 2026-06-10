"""Regression tests for LCP-family public API guards and invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix

_STATE_E = 1
_STATE_U = 2
_TIME_COLS = ["1", "2", "3", "4", "5", "6"]

_LCP_CPP_CLASSES = [
    "LCPdistance",
    "LCPspellDistance",
    "LCPmstDistance",
    "LCPprodDistance",
]


def _sequence_data(rows: list[list[int]], ids: list[int]) -> SequenceData:
    raw = pd.DataFrame(rows, columns=_TIME_COLS)
    raw.insert(0, "id", ids)
    return SequenceData(
        raw,
        time=_TIME_COLS,
        id_col="id",
        states=[_STATE_E, _STATE_U],
    )


def _appendix_sequences_9_10() -> SequenceData:
    return _sequence_data(
        [
            [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
            [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
        ],
        [9, 10],
    )


def _pair_distance(seqdata: SequenceData, method: str, **kwargs) -> float:
    dist = get_distance_matrix(seqdata, method=method, norm="none", **kwargs)
    return float(dist.values[0, 1])


@pytest.fixture
def c_code():
    pytest.importorskip("sequenzo.dissimilarity_measures.c_code")
    import sequenzo.dissimilarity_measures.c_code as module

    return module


def _spell_arrays():
    sequences = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int32)
    durations = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]], dtype=np.float64)
    seqlength = np.array([3, 3], dtype=np.int32)
    totaldur = np.array([6.0, 6.0], dtype=np.float64)
    return sequences, durations, seqlength, totaldur


def _make_lcp_object(c_code, class_name: str, refseq=None):
    sequences, durations, seqlength, totaldur = _spell_arrays()
    if refseq is None:
        refseq = np.array([-1, -1], dtype=np.int32)

    if class_name == "LCPdistance":
        pos = np.arange(8, dtype=np.int32).reshape(2, 4)
        return c_code.LCPdistance(pos, 0, 1, refseq)
    if class_name == "LCPspellDistance":
        return c_code.LCPspellDistance(
            sequences, durations, seqlength, 0, 1, refseq, 1.0, 6.0
        )
    if class_name == "LCPmstDistance":
        return c_code.LCPmstDistance(
            sequences, durations, seqlength, totaldur, 0, 1, refseq
        )
    if class_name == "LCPprodDistance":
        return c_code.LCPprodDistance(
            sequences, durations, seqlength, totaldur, 0, 1, refseq
        )
    raise AssertionError(f"unknown class {class_name}")


@pytest.mark.parametrize("method", ["LCPmst", "RLCPmst", "LCPprod", "RLCPprod"])
def test_lcp_elzinga_variants_reject_nondefault_tpow(method):
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="tpow"):
        get_distance_matrix(seqdata, method=method, norm="none", tpow=2.0)


@pytest.mark.parametrize("method", ["LCPprod", "RLCPprod"])
@pytest.mark.parametrize("norm", ["maxdist", "gmean", "YujianBo", "maxlength"])
def test_lcpprod_rejects_normalization(method, norm):
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="norm"):
        get_distance_matrix(seqdata, method=method, norm=norm)


def test_lcpprod_auto_norm_still_allowed():
    seqdata = _appendix_sequences_9_10()
    dist = get_distance_matrix(seqdata, method="LCPprod", norm="auto")
    assert float(dist.values[0, 1]) == pytest.approx(-8.0)


def test_lcpspell_maxdist_rejects_duration_ref_below_observed_spell_max():
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="duration_ref"):
        get_distance_matrix(
            seqdata,
            method="LCPspell",
            norm="maxdist",
            expcost=1.0,
            duration_ref=2.0,
        )


def test_lcpspell_auto_norm_rejects_duration_ref_below_observed_spell_max():
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="duration_ref"):
        get_distance_matrix(
            seqdata,
            method="LCPspell",
            norm="auto",
            expcost=1.0,
            duration_ref=2.0,
        )


def test_lcpspell_none_norm_allows_small_duration_ref():
    seqdata = _appendix_sequences_9_10()
    with pytest.warns(UserWarning, match="duration_ref"):
        raw = get_distance_matrix(
            seqdata,
            method="LCPspell",
            norm="none",
            expcost=1.0,
            duration_ref=2.0,
        )
    assert float(raw.values[0, 1]) == pytest.approx(2.0)


def test_lcpmst_gmean_is_bounded_when_tx_ne_ty():
    seqdata = _sequence_data(
        [
            [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
            [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
        ],
        [1, 2],
    )
    dist = get_distance_matrix(seqdata, method="LCPmst", norm="gmean")
    values = dist.to_numpy(dtype=np.float64)
    assert np.all(values[np.isfinite(values)] >= 0.0)
    assert np.all(values[np.isfinite(values)] <= 1.0)


def test_lcp_cpp_kernel_handles_noncontiguous_input(c_code):
    base = np.arange(12, dtype=np.int32).reshape(3, 4)
    noncontiguous = base[:, ::-1]
    refseq = np.array([-1, -1], dtype=np.int32)

    dist_obj = c_code.LCPdistance(noncontiguous, 0, 1, refseq)
    matrix = np.asarray(dist_obj.compute_all_distances(), dtype=np.float64)
    expected = c_code.LCPdistance(np.ascontiguousarray(base[:, ::-1]), 0, 1, refseq)
    expected_matrix = np.asarray(expected.compute_all_distances(), dtype=np.float64)
    np.testing.assert_allclose(matrix, expected_matrix)


def test_rlcp_positionwise_treats_zero_as_regular_state(c_code):
    refseq = np.array([-1, -1], dtype=np.int32)
    with_shared_zero = np.array(
        [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]],
        dtype=np.int32,
    )
    without_shared_zero = np.array(
        [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]],
        dtype=np.int32,
    )

    d_shared = float(
        np.asarray(c_code.LCPdistance(with_shared_zero, 0, -1, refseq).compute_all_distances())[0, 1]
    )
    d_not_shared = float(
        np.asarray(
            c_code.LCPdistance(without_shared_zero, 0, -1, refseq).compute_all_distances()
        )[0, 1]
    )
    assert d_shared < d_not_shared


def test_rlcp_spell_ignores_columns_beyond_seqlength(c_code):
    refseq = np.array([-1, -1], dtype=np.int32)
    seqlength = np.array([3, 4], dtype=np.int32)
    durations = np.ones((2, 5), dtype=np.float64)
    totaldur = np.array([3.0, 4.0], dtype=np.float64)

    sequences_a = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=np.int32)
    sequences_b = np.array([[1, 2, 3, 0, 9], [4, 5, 6, 7, 9]], dtype=np.int32)

    dist_a = float(
        np.asarray(
            c_code.LCPmstDistance(
                sequences_a, durations, seqlength, totaldur, 0, -1, refseq
            ).compute_all_distances()
        )[0, 1]
    )
    dist_b = float(
        np.asarray(
            c_code.LCPmstDistance(
                sequences_b, durations, seqlength, totaldur, 0, -1, refseq
            ).compute_all_distances()
        )[0, 1]
    )
    assert dist_a == pytest.approx(dist_b)


def test_lcpprod_cpp_rejects_nonzero_norm(c_code):
    sequences, durations, seqlength, totaldur = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="norm=none"):
        c_code.LCPprodDistance(
            sequences, durations, seqlength, totaldur, 3, 1, refseq
        )


@pytest.mark.parametrize("class_name", _LCP_CPP_CLASSES)
def test_lcp_cpp_refseq_distances_reject_missing_reference(c_code, class_name):
    dist_obj = _make_lcp_object(c_code, class_name)
    with pytest.raises(ValueError, match="reference sequence"):
        dist_obj.compute_refseq_distances()


def test_lcpmst_cpp_rejects_inconsistent_totaldur(c_code):
    sequences, durations, seqlength, _ = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="totaldur"):
        c_code.LCPmstDistance(
            sequences, durations, seqlength, np.array([100.0, 6.0]), 0, 1, refseq
        )


def test_lcpmst_cpp_rejects_zero_active_duration(c_code):
    sequences = np.array([[1, 2], [1, 2]], dtype=np.int32)
    durations = np.array([[0.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    seqlength = np.array([2, 2], dtype=np.int32)
    totaldur = np.array([2.0, 3.0], dtype=np.float64)
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="strictly positive"):
        c_code.LCPmstDistance(
            sequences, durations, seqlength, totaldur, 0, 1, refseq
        )


def test_lcpspell_cpp_maxdist_rejects_small_duration_ref(c_code):
    sequences, durations, seqlength, _ = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="duration_ref"):
        c_code.LCPspellDistance(
            sequences, durations, seqlength, 3, 1, refseq, 1.0, 2.0
        )


def test_lcp_cpp_rejects_invalid_norm_code(c_code):
    sequences = np.arange(12, dtype=np.int32).reshape(3, 4)
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="norm must be one of"):
        c_code.LCPdistance(sequences, 99, 1, refseq)


def test_lcp_cpp_rejects_out_of_range_reference(c_code):
    sequences = np.arange(8, dtype=np.int32).reshape(2, 4)
    refseq = np.array([5, 5], dtype=np.int32)
    with pytest.raises(ValueError, match="out of bounds"):
        c_code.LCPdistance(sequences, 0, 1, refseq)


@pytest.mark.parametrize("class_name", _LCP_CPP_CLASSES)
@pytest.mark.parametrize("i,j", [(-1, 0), (0, -1), (99, 0), (0, 99)])
def test_lcp_cpp_compute_distance_rejects_out_of_range_indices(c_code, class_name, i, j):
    dist_obj = _make_lcp_object(c_code, class_name)
    with pytest.raises((ValueError, IndexError), match="out of bounds"):
        dist_obj.compute_distance(i, j)


@pytest.mark.parametrize("norm", [1, 2, 4])
def test_lcpspell_cpp_rejects_unsupported_normalization(c_code, norm):
    sequences, durations, seqlength, _ = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="LCPspell"):
        c_code.LCPspellDistance(
            sequences, durations, seqlength, norm, 1, refseq, 1.0, 6.0
        )


def test_lcpspell_cpp_maxdist_accepts_duration_ref_equal_to_max_duration(c_code):
    sequences, durations, seqlength, _ = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    obj = c_code.LCPspellDistance(
        sequences, durations, seqlength, 3, 1, refseq, 1.0, 3.0
    )
    matrix = np.asarray(obj.compute_all_distances(), dtype=np.float64)
    assert np.all(matrix[np.isfinite(matrix)] >= 0.0)
    assert np.all(matrix[np.isfinite(matrix)] <= 1.0 + 1e-12)


def test_lcpspell_cpp_maxdist_handles_two_empty_sequences(c_code):
    sequences = np.zeros((2, 1), dtype=np.int32)
    durations = np.zeros((2, 1), dtype=np.float64)
    seqlength = np.array([0, 0], dtype=np.int32)
    refseq = np.array([-1, -1], dtype=np.int32)

    obj = c_code.LCPspellDistance(
        sequences, durations, seqlength, 3, 1, refseq, 1.0, 6.0
    )
    matrix = np.asarray(obj.compute_all_distances(), dtype=np.float64)

    assert matrix[0, 1] == pytest.approx(0.0)
    assert np.all(np.isfinite(matrix))


def test_lcpmst_cpp_rejects_totaldur_when_active_sum_overflows(c_code):
    sequences = np.array([[1, 2]], dtype=np.int32)
    huge = 1.0e308
    durations = np.array([[huge, huge]], dtype=np.float64)
    seqlength = np.array([2], dtype=np.int32)
    totaldur = np.array([np.inf], dtype=np.float64)
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="finite"):
        c_code.LCPmstDistance(
            sequences, durations, seqlength, totaldur, 0, 1, refseq
        )


def test_lcpmst_wrapper_rejects_maxlength_norm():
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="maxlength"):
        get_distance_matrix(seqdata, method="LCPmst", norm="maxlength")


@pytest.mark.parametrize("norm", ["maxlength", "gmean", "YujianBo"])
def test_lcpspell_wrapper_rejects_unsupported_normalization(norm):
    seqdata = _appendix_sequences_9_10()
    with pytest.raises(ValueError, match="norm"):
        get_distance_matrix(
            seqdata,
            method="LCPspell",
            norm=norm,
            expcost=1.0,
            duration_ref=6.0,
        )


def test_lcpmst_cpp_rejects_maxlength_norm(c_code):
    sequences, durations, seqlength, totaldur = _spell_arrays()
    refseq = np.array([-1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="maxlength"):
        c_code.LCPmstDistance(
            sequences, durations, seqlength, totaldur, 1, 1, refseq
        )


@pytest.mark.parametrize("class_name", ["LCPmstDistance", "LCPprodDistance", "LCPspellDistance"])
def test_spell_cpp_rejects_adjacent_duplicate_active_states(c_code, class_name):
    sequences = np.array([[1, 1, 2], [1, 2, 1]], dtype=np.int32)
    durations = np.array([[2.0, 3.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float64)
    seqlength = np.array([3, 3], dtype=np.int32)
    totaldur = np.array([6.0, 6.0], dtype=np.float64)
    refseq = np.array([-1, -1], dtype=np.int32)

    with pytest.raises(ValueError, match="distinct"):
        if class_name == "LCPspellDistance":
            c_code.LCPspellDistance(
                sequences, durations, seqlength, 0, 1, refseq, 0.0, 6.0
            )
        elif class_name == "LCPmstDistance":
            c_code.LCPmstDistance(
                sequences, durations, seqlength, totaldur, 0, 1, refseq
            )
        else:
            c_code.LCPprodDistance(
                sequences, durations, seqlength, totaldur, 0, 1, refseq
            )


def test_lcp_cpp_pairwise_refseq_shape(c_code):
    sequences = np.arange(16, dtype=np.int32).reshape(4, 4)
    refseq = np.array([-1, -1], dtype=np.int32)
    matrix = np.asarray(c_code.LCPdistance(sequences, 0, 1, refseq).compute_all_distances())
    assert matrix.shape == (4, 4)


def test_lcp_cpp_single_reference_refseq_shape_and_values(c_code):
    sequences = np.arange(12, dtype=np.int32).reshape(3, 4)
    refseq = np.array([2, 2], dtype=np.int32)
    full = np.asarray(c_code.LCPdistance(sequences, 0, 1, np.array([-1, -1], np.int32)).compute_all_distances())
    refdist = np.asarray(c_code.LCPdistance(sequences, 0, 1, refseq).compute_refseq_distances())
    assert refdist.shape == (3, 1)
    np.testing.assert_allclose(refdist[:, 0], full[:, 1])


def test_lcp_cpp_subset_refseq_shape_and_values(c_code):
    sequences = np.arange(20, dtype=np.int32).reshape(5, 4)
    refseq = np.array([2, 5], dtype=np.int32)
    block = np.asarray(
        c_code.LCPdistance(sequences, 0, 1, np.array([-1, -1], np.int32)).compute_all_distances()
    )[:2, 2:]
    refdist = np.asarray(c_code.LCPdistance(sequences, 0, 1, refseq).compute_refseq_distances())
    assert refdist.shape == (2, 3)
    np.testing.assert_allclose(refdist, block)


def test_lcp_cpp_rejects_invalid_subset_refseq_range(c_code):
    sequences = np.arange(12, dtype=np.int32).reshape(3, 4)
    with pytest.raises(ValueError, match="subset mode"):
        c_code.LCPdistance(sequences, 0, 1, np.array([0, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="subset mode"):
        c_code.LCPdistance(sequences, 0, 1, np.array([1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="Invalid refseqS"):
        c_code.LCPdistance(sequences, 0, 1, np.array([3, 2], dtype=np.int32))


def test_lcpmst_cpp_padding_duration_beyond_seqlength_allowed(c_code):
    sequences = np.array([[1, 2, 3]], dtype=np.int32)
    durations = np.array([[1.0, 2.0, -999.0]], dtype=np.float64)
    seqlength = np.array([2], dtype=np.int32)
    totaldur = np.array([3.0], dtype=np.float64)
    refseq = np.array([-1, -1], dtype=np.int32)
    obj = c_code.LCPmstDistance(sequences, durations, seqlength, totaldur, 0, 1, refseq)
    assert obj.compute_distance(0, 0) == pytest.approx(0.0)


def test_lcpspell_unit_recoding_invariance(c_code):
    sequences = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int32)
    durations_yearly = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]], dtype=np.float64)
    durations_monthly = durations_yearly * 12.0
    seqlength = np.array([3, 3], dtype=np.int32)
    refseq = np.array([-1, -1], dtype=np.int32)

    yearly = c_code.LCPspellDistance(
        sequences, durations_yearly, seqlength, 0, 1, refseq, 1.0, 6.0
    )
    monthly = c_code.LCPspellDistance(
        sequences, durations_monthly, seqlength, 0, 1, refseq, 1.0, 72.0
    )
    yearly_dist = float(np.asarray(yearly.compute_all_distances())[0, 1])
    monthly_dist = float(np.asarray(monthly.compute_all_distances())[0, 1])
    assert yearly_dist == pytest.approx(2.0 / 3.0)
    assert monthly_dist == pytest.approx(yearly_dist)
