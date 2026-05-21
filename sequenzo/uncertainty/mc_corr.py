"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_corr.py
@Time    : 20/05/2026 10:05
@Desc    : Correlation of observed vs replicate distance matrices.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform

from .mc_diss import MCDissList


def _weighted_corr(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    method: str = "Spearman",
) -> float:
    """Weighted Pearson/Spearman correlation (approximation of R ``wCorr::weightedCorr``)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if method.lower() == "spearman":
        x = stats.rankdata(x, method="average")
        y = stats.rankdata(y, method="average")
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def _pairwise_weights(n: int, weights: np.ndarray) -> np.ndarray:
    """Condensed weight vector for pairwise observations (R ``tcrossprod(weights)``)."""
    outer = np.outer(weights, weights)
    return squareform(outer, checks=False)


def mc_disscorr(
    disslist: MCDissList,
    diss_o: Optional[Union[np.ndarray, np.ndarray]] = None,
    *,
    method: str = "Spearman",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Correlation between observed and each MC-replicated distance matrix (R ``MCdisscorr``).
    """
    toref = getattr(disslist, "toref", False)
    items = list(disslist)
    n = len(items)
    if diss_o is None:
        diss_o = items[-1]
        items = items[:-1]
        n = n - 1

    diss_o_mat = np.asarray(diss_o, dtype=float)
    if diss_o_mat.ndim == 1:
        diss_o_mat = squareform(diss_o_mat)

    ncases = diss_o_mat.shape[0]
    if weights is None:
        weights = np.ones(ncases, dtype=float)
    w_pair = _pairwise_weights(ncases, weights)

    corrs = []
    for mat in items:
        m = np.asarray(mat, dtype=float)
        if m.ndim == 1:
            m = squareform(m)
        if toref:
            raise NotImplementedError("toref=TRUE not yet implemented for MCdisscorr")
        o_vec = squareform(diss_o_mat, checks=False)
        m_vec = squareform(m, checks=False)
        corrs.append(_weighted_corr(m_vec, o_vec, w_pair, method))
    return np.array(corrs, dtype=float)


def _wcmdscale_one(
    diss: np.ndarray,
    weights: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """First weighted classical MDS axis (R ``vegan::wcmdscale`` approximation)."""
    d = np.asarray(diss, dtype=float)
    if d.ndim == 1:
        d = squareform(d)
    n = d.shape[0]
    d2 = d ** 2
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / w.sum()
    center = np.eye(n) - np.outer(w, np.ones(n))
    b = -0.5 * center @ d2 @ center
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    lam = max(eigvals[0], 0.0)
    if lam <= 0:
        return np.zeros(n, dtype=float)
    return eigvecs[:, 0] * np.sqrt(lam)


def mc_mdscorr(
    disslist: MCDissList,
    diss_o: Optional[Union[np.ndarray, np.ndarray]] = None,
    *,
    method: str = "Spearman",
    weights: Optional[np.ndarray] = None,
    what: str = "corr",
) -> Union[np.ndarray, dict]:
    """
    Correlation of first MDS factor: observed vs MC-replicated distances (R ``MCmdscorr``).
    """
    items = list(disslist)
    n = len(items)
    if diss_o is None:
        diss_o = items[-1]
        items = items[:-1]
        n = n - 1

    if diss_o.ndim == 1:
        diss_o = squareform(diss_o)
    ncases = diss_o.shape[0]
    if weights is None:
        weights = np.ones(ncases, dtype=float)

    mds_o = _wcmdscale_one(diss_o, weights, k=1)
    mds_list = [_wcmdscale_one(m, weights, k=1) for m in items]

    if what == "mds":
        mds_list.append(mds_o)
        return {"mdslist": mds_list, "mds.o": mds_o}

    corrs = np.array(
        [_weighted_corr(m, mds_o, weights, method) for m in mds_list],
        dtype=float,
    )
    for i in np.where(corrs < 0)[0]:
        corrs[i] = -corrs[i]
        mds_list[i] = -mds_list[i]

    if what == "both":
        return {"corr": corrs, "mdslist": mds_list + [mds_o]}
    return corrs
