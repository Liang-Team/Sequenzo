"""
Appendix regression: Sequences 9 and 10 (ten-sequence illustration).

Validates raw distances against the worked examples in:
  - app:om-omspell-illustration (OMspellUnitFree, tau=6)
  - app:lcp-variants (LCP, LCPmst, LCPprod, LCPspell)

Seq. 9 (time-expanded): E E E U E E
Seq. 10 (time-expanded): E U U U E E
Spell form: (E,3)(U,1)(E,2) vs (E,1)(U,3)(E,2); T_x = T_y = 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix

# State codes: E = 1, U = 2 (matches appendix labels).
_STATE_E = 1
_STATE_U = 2
_TAU = 6.0


def _appendix_sequences_9_10() -> SequenceData:
    """Build SequenceData for appendix Sequences 9 and 10 only."""
    time_cols = ["1", "2", "3", "4", "5", "6"]
    raw = pd.DataFrame(
        [
            [_STATE_E, _STATE_E, _STATE_E, _STATE_U, _STATE_E, _STATE_E],
            [_STATE_E, _STATE_U, _STATE_U, _STATE_U, _STATE_E, _STATE_E],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", [9, 10])
    return SequenceData(
        raw,
        time=time_cols,
        id_col="id",
        states=[_STATE_E, _STATE_U],
    )


def _pair_distance(seqdata: SequenceData, method: str, **kwargs) -> float:
    """Off-diagonal distance between sequence index 0 (Seq. 9) and 1 (Seq. 10)."""
    dist = get_distance_matrix(seqdata, method=method, norm="none", **kwargs)
    return float(dist.values[0, 1])


@pytest.fixture(scope="module")
def appendix_seqdata():
    return _appendix_sequences_9_10()


@pytest.fixture(scope="module")
def om_substitution_matrix():
    """c_indel = 1; sigma(E, U) = 2 (appendix OM illustration)."""
    return np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)


class TestAppendixSequences9And10:
    """Raw distances; norm='none' throughout."""

    def test_lcp_position_wise_raw_is_10(self, appendix_seqdata):
        assert _pair_distance(appendix_seqdata, "LCP") == pytest.approx(10.0)

    def test_lcpmst_dss_raw_is_4(self, appendix_seqdata):
        assert _pair_distance(appendix_seqdata, "LCPmst") == pytest.approx(4.0)

    def test_lcpprod_dss_raw_is_negative_8(self, appendix_seqdata):
        assert _pair_distance(appendix_seqdata, "LCPprod") == pytest.approx(-8.0)

    def test_lcpspell_expcost_zero_raw_is_0(self, appendix_seqdata):
        assert _pair_distance(
            appendix_seqdata,
            "LCPspell",
            expcost=0.0,
            duration_ref=_TAU,
        ) == pytest.approx(0.0)

    def test_lcpspell_expcost_one_tau_six_raw_is_two_thirds(self, appendix_seqdata):
        assert _pair_distance(
            appendix_seqdata,
            "LCPspell",
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0 / 3.0)

    def test_omspell_unit_free_expcost_one_tau_six_raw_is_two_thirds(
        self, appendix_seqdata, om_substitution_matrix
    ):
        assert _pair_distance(
            appendix_seqdata,
            "OMspellUnitFree",
            sm=om_substitution_matrix,
            indel=1.0,
            expcost=1.0,
            duration_ref=_TAU,
        ) == pytest.approx(2.0 / 3.0)
