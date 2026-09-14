"""Ordinary LRT must not rewrite sampling zeros as structural zeros."""

from __future__ import annotations

import numpy as np
import pytest

from sequenzo.multidomain.association_between_domains import _log_likelihood_ratio_test


def test_lrt_keeps_sampling_zeros_and_ordinary_df():
    xtab = np.array([[10.0, 0.0], [0.0, 10.0]])
    lrt, df, _p = _log_likelihood_ratio_test(xtab)
    assert df == 1
    # 2 * (10 log(10/5) + 10 log(10/5)) = 40 log 2
    assert lrt == pytest.approx(40.0 * np.log(2.0))


def test_struct_zero_without_mask_does_not_rewrite_counts(recwarn):
    xtab = np.array([[10.0, 0.0], [0.0, 10.0]])
    lrt_plain, df_plain, _ = _log_likelihood_ratio_test(xtab, struct_zero=False)
    lrt_flag, df_flag, _ = _log_likelihood_ratio_test(xtab, struct_zero=True)
    assert df_plain == df_flag == 1
    assert lrt_plain == pytest.approx(lrt_flag)


def test_explicit_structural_zero_mask_reduces_df():
    xtab = np.array([[5.0, 1.0, 0.0], [2.0, 4.0, 1.0]])
    mask = np.array([[False, False, True], [False, False, False]])
    _lrt, df, _p = _log_likelihood_ratio_test(
        xtab, structural_zero_mask=mask
    )
    assert df == (2 - 1) * (3 - 1) - 1
