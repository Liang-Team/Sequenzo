"""
Paper appendix: worked distances for illustration sequences (main.tex).

Labels:
  - app:om-omspell-illustration — Sequences 9--10 (same spell order, duration mismatch)
                          and Sequences 1--2 (different-state single-spell substitution)
  - app:lcp-variants — LCP, LCPmst, LCPprod, LCPspell for both pairs

Common settings: c_indel = 1, sigma(E,U) = 2, lambda = 1 (expcost), tau = 6 for
reference-scaled measures. State codes: E = 1, U = 2.

Original OMspell uses Studer & Ritschard (2016) expansion (d-1), (d_i+d_j-2).
OMspellRS uses full reference-scaled duration (d/tau, (d_i+d_j)/tau).
These tests are the regression target (not TraMineR seqdist parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix

_STATE_E = 1
_STATE_U = 2
_TAU = 6.0
_TIME_COLS = ["1", "2", "3", "4", "5", "6"]


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
    """Seq. 9: E E E U E E; Seq. 10: E U U U E E."""
    return _sequence_data(
        [
            [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
            [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
        ],
        [9, 10],
    )


def _appendix_sequences_1_2() -> SequenceData:
    """Seq. 1: U x6; Seq. 2: E x6 (different-state single-spell pair)."""
    return _sequence_data(
        [
            [_STATE_U] * 6,
            [_STATE_E] * 6,
        ],
        [1, 2],
    )


def _pair_distance(
    seqdata: SequenceData,
    method: str,
    *,
    norm: str = "none",
    **kwargs,
) -> float:
    """Off-diagonal distance between row 0 and row 1."""
    dist = get_distance_matrix(seqdata, method=method, norm=norm, **kwargs)
    return float(dist.values[0, 1])


@pytest.fixture(scope="module")
def appendix_seqdata_9_10():
    return _appendix_sequences_9_10()


@pytest.fixture(scope="module")
def appendix_seqdata_1_2():
    return _appendix_sequences_1_2()


@pytest.fixture(scope="module")
def om_substitution_matrix():
    """c_indel = 1; sigma(E, U) = 2."""
    return np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)


class TestAppendixSequences9And10:
    """Same spell order (E->U->E), different durations; tau = 6."""

    def test_om_position_wise_raw_is_4(self, appendix_seqdata_9_10, om_substitution_matrix):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "OM",
            sm=om_substitution_matrix,
            indel=1.0,
        ) == pytest.approx(4.0)

    def test_lcp_position_wise_raw_is_10(self, appendix_seqdata_9_10):
        assert _pair_distance(appendix_seqdata_9_10, "LCP") == pytest.approx(10.0)

    def test_lcpmst_dss_raw_is_4(self, appendix_seqdata_9_10):
        assert _pair_distance(appendix_seqdata_9_10, "LCPmst") == pytest.approx(4.0)

    def test_lcpprod_dss_raw_is_negative_8(self, appendix_seqdata_9_10):
        assert _pair_distance(appendix_seqdata_9_10, "LCPprod") == pytest.approx(-8.0)

    def test_lcpspell_expcost_zero_raw_is_0(self, appendix_seqdata_9_10):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "LCPspell",
            expcost=0.0,
            duration_ref=_TAU,
        ) == pytest.approx(0.0)

    def test_lcpspell_expcost_one_tau_six_raw_is_two_thirds(self, appendix_seqdata_9_10):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "LCPspell",
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0 / 3.0)

    def test_lcpspell_maxdist_norm_is_two_twenty_sevenths(self, appendix_seqdata_9_10):
        """Appendix: raw 2/3, d_max = 9 -> 2/27 under maxdist."""
        assert _pair_distance(
            appendix_seqdata_9_10,
            "LCPspell",
            norm="maxdist",
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0 / 27.0)

    def test_omspell_expcost_one_raw_is_4(self, appendix_seqdata_9_10, om_substitution_matrix):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "OMspell",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
        ) == pytest.approx(4.0)

    def test_omspell_expcost_half_raw_is_2(self, appendix_seqdata_9_10, om_substitution_matrix):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "OMspell",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=0.5,
        ) == pytest.approx(2.0)

    def test_omspell_unit_free_expcost_one_tau_six_raw_is_two_thirds(
        self, appendix_seqdata_9_10, om_substitution_matrix
    ):
        assert _pair_distance(
            appendix_seqdata_9_10,
            "OMspellRS",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0 / 3.0)

    def test_omspell_unit_free_auto_norm_uses_yujian_bo(
        self, appendix_seqdata_9_10, om_substitution_matrix
    ):
        d_auto = _pair_distance(
            appendix_seqdata_9_10,
            "OMspellRS",
            norm="auto",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=_TAU,
        )
        d_yb = _pair_distance(
            appendix_seqdata_9_10,
            "OMspellRS",
            norm="YujianBo",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=_TAU,
        )
        assert d_auto == pytest.approx(d_yb)
        assert d_auto == pytest.approx(2.0 / 13.0)


class TestAppendixSequences1And2:
    """Single-spell pair (U,6) vs (E,6): different-state substitution; d_a+d_b-2 in OMspell."""

    def test_om_position_wise_raw_is_12(self, appendix_seqdata_1_2, om_substitution_matrix):
        assert _pair_distance(
            appendix_seqdata_1_2,
            "OM",
            sm=om_substitution_matrix,
            indel=1.0,
        ) == pytest.approx(12.0)

    def test_lcp_position_wise_raw_is_12(self, appendix_seqdata_1_2):
        assert _pair_distance(appendix_seqdata_1_2, "LCP") == pytest.approx(12.0)

    def test_lcpmst_dss_raw_is_12(self, appendix_seqdata_1_2):
        assert _pair_distance(appendix_seqdata_1_2, "LCPmst") == pytest.approx(12.0)

    def test_lcpprod_dss_raw_is_12(self, appendix_seqdata_1_2):
        assert _pair_distance(appendix_seqdata_1_2, "LCPprod") == pytest.approx(12.0)

    def test_lcpspell_raw_is_2(self, appendix_seqdata_1_2):
        """L=0 spell prefix; n+m-2L = 2 (duration term zero with no matched spells)."""
        assert _pair_distance(
            appendix_seqdata_1_2,
            "LCPspell",
            expcost=0.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0)
        assert _pair_distance(
            appendix_seqdata_1_2,
            "LCPspell",
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0)

    def test_omspell_expcost_one_raw_is_12(self, appendix_seqdata_1_2, om_substitution_matrix):
        """sigma + lambda(d_a+d_b-2) = 2 + (6+6-2) = 12."""
        assert _pair_distance(
            appendix_seqdata_1_2,
            "OMspell",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
        ) == pytest.approx(12.0)

    def test_omspell_unit_free_expcost_one_raw_is_4(
        self, appendix_seqdata_1_2, om_substitution_matrix
    ):
        """sigma + lambda(d_a+d_b)/tau = 2 + 12/6 = 4."""
        assert _pair_distance(
            appendix_seqdata_1_2,
            "OMspellRS",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(4.0)
