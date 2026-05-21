"""
@Author  : Yuqi Liang 梁彧祺
@File    : _params.py
@Time    : 02/05/2026 08:35
@Desc    : Parse timing-error and alteration-model parameters (MCseqReplic parity).
"""
from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

ModelName = Literal["keep.dss", "indep", "relative"]

MODEL_TO_METHOD = {
    "keep.dss": 1,
    "indep": 2,
    "relative": 3,
}


def parse_model(model: str) -> int:
    """Map model name to internal method code (1=keep.dss, 2=indep, 3=relative)."""
    if model not in MODEL_TO_METHOD:
        raise ValueError(f"Unknown model value: {model!r}")
    return MODEL_TO_METHOD[model]


def parse_kchanges(
    kchanges: Union[int, str, None],
) -> Union[int, None]:
    """
    Parse ``kchanges`` like R MCseqReplic.

    ``None`` -> alter all transitions; integer -> at most that many;
    ``"rand"`` -> random count per sequence (stored as -1 internally).
    """
    if kchanges is None:
        return None
    if isinstance(kchanges, str):
        if kchanges == "rand":
            return -1
        raise ValueError("Invalid kchanges value!")
    return int(kchanges)


def parse_jprob(
    J: Union[int, float, np.ndarray, list],
) -> Tuple[Union[int, np.ndarray], np.ndarray]:
    """
    Return ``(J_scalar_or_K, Jprob)`` matching R ``MCseqReplicate`` / ``seqdistMCSE``.

    If ``J`` is a scalar integer ``K``, ``Jprob`` is the integer ``K`` passed to ``move()``.
    If ``J`` is a probability vector, it is normalized and ``K = (len(J)-1)//2``.
    """
    if np.isscalar(J) or isinstance(J, (int, float)):
        k = int(J)
        return k, k
    j_arr = np.asarray(J, dtype=float).ravel()
    if j_arr.size % 2 == 0:
        raise ValueError("Length of J must be odd!")
    if (j_arr < 0).any():
        raise ValueError("Negative values in J")
    jprob = j_arr / j_arr.sum()
    k = (j_arr.size - 1) // 2
    return k, jprob
