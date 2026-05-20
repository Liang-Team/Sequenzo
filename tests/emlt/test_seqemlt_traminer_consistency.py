"""
Numerical consistency tests: Sequenzo ``compute_emlt`` vs TraMineRextras ``seqemlt``.

Generate reference files first:

    Rscript tests/emlt/traminer_reference_seqemlt.R
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.datasets import load_dataset
from sequenzo.emlt import compute_emlt

RTOL = 1e-6
ATOL = 1e-8

REF_DIR = os.path.dirname(__file__)


@pytest.fixture
def lsog_seqdata():
    df = load_dataset("dyadic_children")
    time_list = sorted([c for c in df.columns if str(c).isdigit()], key=int)
    df = df.head(20)
    return SequenceData(df, time=time_list, id_col="dyadID", states=[1, 2, 3, 4, 5, 6])


def _ref_path(name: str) -> str:
    return os.path.join(REF_DIR, name)


def _load_matrix_values(name: str) -> np.ndarray | None:
    """
    Load the numeric block from a reference CSV.

    Row/column labels such as ``1.10`` are not used: pandas would parse them as
    floats and duplicate indices (e.g. ``1.1`` for times 1 and 10).
    """
    path = _ref_path(name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.iloc[:, 1:].to_numpy(dtype=float)


def _assert_matrix_close(sequenzo_df: pd.DataFrame, ref_values: np.ndarray, label: str) -> None:
    s = sequenzo_df.to_numpy(dtype=float)
    r = ref_values
    assert s.shape == r.shape, f"{label}: shape mismatch"
    mask = ~(np.isnan(s) | np.isnan(r))
    if mask.any():
        np.testing.assert_allclose(s[mask], r[mask], rtol=RTOL, atol=ATOL, err_msg=label)
    nan_s = np.isnan(s) & ~np.isnan(r)
    nan_r = np.isnan(r) & ~np.isnan(s)
    assert not nan_s.any() and not nan_r.any(), f"{label}: NA pattern mismatch"


def _assert_vector_close(sequenzo: pd.Series, ref_values: np.ndarray, label: str) -> None:
    ref = ref_values.ravel()
    assert len(sequenzo) == len(ref), f"{label}: length mismatch"
    np.testing.assert_allclose(
        sequenzo.to_numpy(dtype=float),
        ref,
        rtol=RTOL,
        atol=ATOL,
        err_msg=label,
    )


@pytest.mark.parametrize(
    "tag,a,b,weighted",
    [
        ("weighted_a1_b1", 1.0, 1.0, True),
        ("unweighted_a2_b3", 2.0, 3.0, False),
    ],
)
def test_seqemlt_matches_traminer(lsog_seqdata, tag, a, b, weighted):
    prefix = f"ref_seqemlt_{tag}"
    if _load_matrix_values(f"{prefix}_sit_freq.csv") is None:
        pytest.skip("Reference files not found. Run tests/emlt/traminer_reference_seqemlt.R first.")

    result = compute_emlt(lsog_seqdata, a=a, b=b, weighted=weighted)

    _assert_vector_close(result.sit_freq, _load_matrix_values(f"{prefix}_sit_freq.csv"), "sit.freq")
    _assert_matrix_close(
        result.sit_transrate, _load_matrix_values(f"{prefix}_sit_transrate.csv"), "sit.transrate"
    )
    _assert_matrix_close(result.sit_profil, _load_matrix_values(f"{prefix}_sit_profil.csv"), "sit.profil")
    _assert_matrix_close(result.distance_matrix, _load_matrix_values(f"{prefix}_c.csv"), "c")
    _assert_matrix_close(
        pd.DataFrame(result.benz_covariance),
        _load_matrix_values(f"{prefix}_d.csv"),
        "d",
    )
    _assert_matrix_close(result.sit_cor, _load_matrix_values(f"{prefix}_sit_cor.csv"), "sit.cor")
    np.testing.assert_allclose(
        result.coord,
        _load_matrix_values(f"{prefix}_coord.csv"),
        rtol=RTOL,
        atol=ATOL,
        err_msg="coord",
    )
    np.testing.assert_allclose(
        result.pca["scores"],
        _load_matrix_values(f"{prefix}_pca_scores.csv"),
        rtol=RTOL,
        atol=ATOL,
        err_msg="pca.scores",
    )
