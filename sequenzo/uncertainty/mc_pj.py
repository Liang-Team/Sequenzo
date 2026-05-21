"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_pj.py
@Time    : 04/05/2026 11:10
@Desc    : Poisson timing-error probability distribution (MCpj).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def mc_pj(
    emean: Union[float, Sequence[float]],
    pzero: Optional[float] = None,
    *,
    maxterr: int = 10,
    pinterv: float = 0.99,
    fill_short_side: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build probability vector ``Pj`` for timing errors (R ``MCpj``).

    Parameters
    ----------
    emean
        Expected size of non-zero timing errors (> 1). Scalar or length-2
        ``(backward, forward)`` means.
    pzero
        Probability of no error in ``[0, 1]``. If ``None``, set from Poisson mass at 0.
    maxterr
        Maximum absolute error size considered.
    pinterv
        Interval width factor for root-finding on ``lambda``.
    fill_short_side
        Pad shorter tail with zeros so both sides have equal length.

    Returns
    -------
    pj
        Probability vector ``(p_{-K}, ..., p_0, ..., p_K)`` summing to 1.
    lambda_
        Estimated Poisson rate(s), shape ``(1,)`` or ``(2,)``.
    """
    emean_arr = np.asarray(emean, dtype=float).ravel()
    if emean_arr.size > 2:
        emean_arr = emean_arr[:2]
    n_emean = 1 if emean_arr.size == 1 else 2
    vemean = emean_arr
    pz = 0.0
    lbda_list = []
    lpois: list[list[float]] = []

    for i in range(n_emean):
        expected = float(vemean[i])
        if expected <= 1.0:
            raise ValueError("Emean must be strictly greater than 1")

        def implicit_fun(lam: float, exp_val: float = expected) -> float:
            return exp_val - lam / (1.0 - np.exp(-lam))

        interv = (expected - expected * pinterv, expected + expected * pinterv)
        try:
            lam_root = brentq(implicit_fun, interv[0], interv[1])
        except ValueError as exc:
            raise ValueError(
                f"Error finding root for Emean={expected}: {exc}. "
                "Try increasing pinterv."
            ) from exc
        lam = max(lam_root, 0.001)
        pois = stats.poisson.pmf(np.arange(1, maxterr + 1), lam)
        pz = max(pz, stats.poisson.pmf(0, lam))
        lbda_list.append(lam)
        tail = pois.tolist()
        if pois.size > int(np.ceil(lam)):
            hend = pois[int(np.ceil(lam)) :]
            hend = hend[hend > 0.005]
            tail = pois[: int(np.ceil(lam))].tolist() + hend.tolist()
        lpois.append(tail)

    if n_emean == 1:
        lpois.append(lpois[0][:])

    if fill_short_side and len(lpois[0]) != len(lpois[1]):
        lg = [len(lpois[0]), len(lpois[1])]
        idmin = int(np.argmin(lg))
        zeros = [0.0] * abs(lg[0] - lg[1])
        lpois[idmin] = lpois[idmin] + zeros

    lpois[0] = list(reversed(lpois[0]))
    if pzero is None:
        pzero = float(pz)
    sumpois = sum(lpois[0]) + sum(lpois[1])
    scale = (1.0 - pzero) / sumpois if sumpois > 0 else 0.0
    left = [p * scale for p in lpois[0]]
    right = [p * scale for p in lpois[1]]
    pj = np.array(left + [pzero] + right, dtype=float)
    lambda_ = np.array(lbda_list, dtype=float)
    return pj, lambda_
