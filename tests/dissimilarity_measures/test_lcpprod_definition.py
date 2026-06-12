"""Regression tests for LCPprod / RLCPprod mathematical definition (Elzinga 2007)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix

_STATE_E = 1
_STATE_U = 2
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


def _pair_distance(seqdata: SequenceData, method: str) -> float:
    dist = get_distance_matrix(seqdata, method=method, norm="none")
    return float(dist.values[0, 1])


def _matrix_distances(seqdata: SequenceData, method: str) -> np.ndarray:
    return get_distance_matrix(seqdata, method=method, norm="none").values


class TestLCPprodIdentity:
    def test_single_spell_identity_is_zero(self):
        seqdata = _sequence_data([[_STATE_E] * 6], [1])
        dist = get_distance_matrix(seqdata, method="LCPprod", norm="none")
        assert float(dist.values[0, 0]) == pytest.approx(0.0)

    def test_multiple_spell_identity_is_zero(self):
        seqdata = _sequence_data(
            [[_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E]],
            [9],
        )
        dist = get_distance_matrix(seqdata, method="LCPprod", norm="none")
        assert float(dist.values[0, 0]) == pytest.approx(0.0)


class TestLCPprodWorkedExamples:
    def test_paper_appendix_sequences_9_10(self):
        seqdata = _sequence_data(
            [
                [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
            ],
            [9, 10],
        )
        assert _pair_distance(seqdata, "LCPprod") == pytest.approx(8.0)

    def test_no_common_prefix(self):
        seqdata = _sequence_data(
            [
                [_STATE_U] * 6,
                [_STATE_E] * 6,
            ],
            [1, 2],
        )
        assert _pair_distance(seqdata, "LCPprod") == pytest.approx(72.0)


class TestRLCPprodSuffix:
    def test_palindromic_spell_order_gives_same_forward_and_reverse_distance(self):
        """E-U-E spell structure: common DSS prefix equals common DSS suffix."""
        seqdata = _sequence_data(
            [
                [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
            ],
            [9, 10],
        )
        assert _pair_distance(seqdata, "LCPprod") == pytest.approx(8.0)
        assert _pair_distance(seqdata, "RLCPprod") == pytest.approx(8.0)

    def test_suffix_only_match(self):
        """(U,2)(E,4) versus (E,6): no DSS prefix, shared terminal E spell."""
        seqdata = _sequence_data(
            [
                [_STATE_U, _STATE_U, _STATE_E, _STATE_E, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_E, _STATE_E, _STATE_E, _STATE_E, _STATE_E],
            ],
            [1, 2],
        )
        assert _pair_distance(seqdata, "LCPprod") == pytest.approx(56.0)
        assert _pair_distance(seqdata, "RLCPprod") == pytest.approx(8.0)


class TestLCPprodInvariants:
    @pytest.fixture
    def three_sequence_panel(self):
        return _sequence_data(
            [
                [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
                [_STATE_U] * 6,
            ],
            [9, 10, 1],
        )

    @pytest.mark.parametrize("method", ["LCPprod", "RLCPprod"])
    def test_diagonal_is_zero(self, three_sequence_panel, method):
        dist = _matrix_distances(three_sequence_panel, method)
        assert np.allclose(np.diag(dist), 0.0, atol=1e-12)

    @pytest.mark.parametrize("method", ["LCPprod", "RLCPprod"])
    def test_symmetry(self, three_sequence_panel, method):
        dist = _matrix_distances(three_sequence_panel, method)
        assert np.allclose(dist, dist.T, atol=1e-12)

    @pytest.mark.parametrize("method", ["LCPprod", "RLCPprod"])
    def test_non_negative_off_diagonal(self, three_sequence_panel, method):
        dist = _matrix_distances(three_sequence_panel, method)
        off_diag = dist[~np.eye(dist.shape[0], dtype=bool)]
        assert np.all(off_diag >= -1e-12)
