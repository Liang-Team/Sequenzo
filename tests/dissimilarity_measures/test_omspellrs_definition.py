"""
Regression tests for OMspellRS reference-scaled duration definition.

OMspellRS uses full spell duration proportions (d/tau, (d_a+d_b)/tau), not the
one-unit-base expansion convention of original OMspell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix

_STATE_E = 1
_STATE_U = 2


def _sequence_data(rows: list[list[int]], ids: list[int], time_cols: list[str]) -> SequenceData:
    raw = pd.DataFrame(rows, columns=time_cols)
    raw.insert(0, "id", ids)
    return SequenceData(
        raw,
        time=time_cols,
        id_col="id",
        states=[_STATE_E, _STATE_U],
    )


def _pair_distance(seqdata: SequenceData, method: str, *, norm: str = "none", **kwargs) -> float:
    dist = get_distance_matrix(seqdata, method=method, norm=norm, **kwargs)
    return float(dist.values[0, 1])


@pytest.fixture(scope="module")
def om_substitution_matrix():
    return np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)


class TestOMspellRSUnitInvariance:
    """Yearly vs monthly coding yields identical OMspellRS distances."""

    def test_different_state_yearly_and_monthly_match(self, om_substitution_matrix):
        yearly = _sequence_data(
            [[_STATE_U] * 6, [_STATE_E] * 6],
            [1, 2],
            [str(i) for i in range(1, 7)],
        )
        monthly_cols = [str(i) for i in range(1, 73)]
        monthly = _sequence_data(
            [[_STATE_U] * 72, [_STATE_E] * 72],
            [1, 2],
            monthly_cols,
        )
        kwargs = dict(
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=6.0,
        )
        d_yearly = _pair_distance(yearly, "OMspellRS", **kwargs)
        kwargs_monthly = dict(kwargs)
        kwargs_monthly["duration_ref"] = 72.0
        d_monthly = _pair_distance(monthly, "OMspellRS", **kwargs_monthly)
        assert d_yearly == pytest.approx(4.0)
        assert d_monthly == pytest.approx(4.0)
        assert d_yearly == pytest.approx(d_monthly)


class TestOMspellRSSameStateSubstitution:
    def test_sequences_9_10_raw_is_two_thirds(self, om_substitution_matrix):
        seqdata = _sequence_data(
            [
                [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
            ],
            [9, 10],
            [str(i) for i in range(1, 7)],
        )
        assert _pair_distance(
            seqdata,
            "OMspellRS",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=6.0,
        ) == pytest.approx(2.0 / 3.0)


class TestOMspellRSYujianBoNormalization:
    def test_sequences_9_10_yb_is_two_thirteenths(self, om_substitution_matrix):
        seqdata = _sequence_data(
            [
                [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
            ],
            [9, 10],
            [str(i) for i in range(1, 7)],
        )
        kwargs = dict(
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=6.0,
        )
        d_yb = _pair_distance(seqdata, "OMspellRS", norm="YujianBo", **kwargs)
        d_auto = _pair_distance(seqdata, "OMspellRS", norm="auto", **kwargs)
        assert d_yb == pytest.approx(2.0 / 13.0)
        assert d_auto == pytest.approx(d_yb)


class TestOMspellBaselineUnchanged:
    def test_sequences_1_2_omspell_raw_is_12(self, om_substitution_matrix):
        seqdata = _sequence_data(
            [[_STATE_U] * 6, [_STATE_E] * 6],
            [1, 2],
            [str(i) for i in range(1, 7)],
        )
        assert _pair_distance(
            seqdata,
            "OMspell",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
        ) == pytest.approx(12.0)


class TestOMspellRSIndelAndDiffStateCosts:
    def test_single_spell_indel_cost_is_c_indel_plus_d_over_tau(self, om_substitution_matrix):
        """Per-spell indel cost: c_indel + lambda*d/tau = 1 + 5/20 for spell (E,5)."""
        time_cols = [str(i) for i in range(1, 6)]
        seqdata = _sequence_data(
            [[_STATE_E] * 5, [_STATE_U] * 5],
            [1, 2],
            time_cols,
        )
        assert _pair_distance(
            seqdata,
            "OMspellRS",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=20.0,
        ) == pytest.approx(2.0 + (5.0 + 5.0) / 20.0)


class TestOMspellRSTpowGuard:
    def test_tpow_not_one_raises_for_omspellrs(self, om_substitution_matrix):
        seqdata = _sequence_data(
            [[_STATE_U] * 6, [_STATE_E] * 6],
            [1, 2],
            [str(i) for i in range(1, 7)],
        )
        with pytest.raises(ValueError, match="tpow"):
            get_distance_matrix(
                seqdata,
                method="OMspellRS",
                sm=om_substitution_matrix,
                indel=1.0,
                expcost=1.0,
                duration_ref=6.0,
                tpow=2.0,
            )

    def test_tpow_two_allowed_for_omspell(self, om_substitution_matrix):
        seqdata = _sequence_data(
            [[_STATE_U] * 6, [_STATE_E] * 6],
            [1, 2],
            [str(i) for i in range(1, 7)],
        )
        get_distance_matrix(
            seqdata,
            method="OMspell",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            tpow=2.0,
        )


class TestOMspellRSParameterGuards:
    @pytest.fixture
    def seqdata(self, om_substitution_matrix):
        del om_substitution_matrix
        return _sequence_data(
            [[_STATE_U] * 6, [_STATE_E] * 6],
            [1, 2],
            [str(i) for i in range(1, 7)],
        )

    @pytest.fixture
    def base_kwargs(self, om_substitution_matrix):
        return dict(
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=6.0,
        )

    def test_duration_ref_zero_raises(self, seqdata, base_kwargs):
        with pytest.raises(ValueError, match="duration_ref"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "duration_ref": 0.0})

    def test_duration_ref_nan_raises(self, seqdata, base_kwargs):
        with pytest.raises(ValueError, match="duration_ref"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "duration_ref": float("nan")})

    def test_expcost_nan_raises(self, seqdata, base_kwargs):
        with pytest.raises(ValueError, match="expcost"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "expcost": float("nan")})

    def test_expcost_negative_raises(self, seqdata, base_kwargs):
        with pytest.raises(ValueError, match="expcost"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "expcost": -1.0})

    def test_negative_indel_raises(self, seqdata, base_kwargs):
        with pytest.raises(ValueError, match="indel"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "indel": -1.0})

    def test_negative_sm_raises(self, seqdata, base_kwargs):
        bad_sm = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="sm"):
            get_distance_matrix(seqdata, method="OMspellRS", **{**base_kwargs, "sm": bad_sm})
