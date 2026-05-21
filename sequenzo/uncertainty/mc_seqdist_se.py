"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_seqdist_se.py
@Time    : 15/05/2026 08:55
@Desc    : MC standard errors over full distance matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from .mc_diss import MCDissList, UDistResult, mc_extract_dist


@dataclass
class DistMCResult:
    """Monte Carlo distance uncertainty results (R class ``distMC``)."""

    mc_mean: Union[np.ndarray, pd.DataFrame]
    mc_sd: Union[np.ndarray, pd.DataFrame]
    n: Optional[int] = None
    r: Optional[int] = None
    weights: Optional[np.ndarray] = None
    diss_o: Optional[Union[np.ndarray, pd.DataFrame]] = None
    mc_bias: Optional[Union[np.ndarray, pd.DataFrame]] = None
    mc_se: Optional[Union[np.ndarray, pd.DataFrame]] = None
    mc_mse: Optional[Union[np.ndarray, pd.DataFrame]] = None
    diss_z: Optional[Union[np.ndarray, pd.DataFrame]] = None
    mc_mean_z: Optional[Union[np.ndarray, pd.DataFrame]] = None
    mean_se: Optional[Union[np.ndarray, pd.DataFrame]] = None
    toref: bool = False
    obs: bool = False
    extra: dict = field(default_factory=dict)


def _strip_r1_labels(labels: List[str]) -> List[str]:
    return [name.replace("R1-", "", 1) if isinstance(name, str) else str(name) for name in labels]


def _matrix_labels(diss, k: int = 1) -> List[str]:
    if isinstance(diss, pd.DataFrame):
        return _strip_r1_labels(list(diss.index))
    if hasattr(diss, "index"):
        return _strip_r1_labels(list(diss.index))
    n = diss.shape[0]
    return [str(i + 1) for i in range(n)]


def mc_seqdist_se(
    dissrepl: Union[str, MCDissList, List, UDistResult],
    mc_r_seqdata: Optional[List] = None,
    *,
    udiss: bool = False,
    full_matrix: bool = False,
    distance_kwargs: Optional[dict] = None,
) -> DistMCResult:
    """
    Mean and SD of dissimilarities across MC-replicated sets (R ``MCseqdistSE``).

    Parameters
    ----------
    dissrepl
        Distance method name, list of distance matrices, or ``UDistResult``.
    mc_r_seqdata
        List of replicated ``SequenceData`` (required when ``dissrepl`` is a method string).
    """
    distance_kwargs = distance_kwargs or {}
    toref = getattr(dissrepl, "toref", False)
    diss_o = None
    n = None

    if isinstance(dissrepl, str):
        if mc_r_seqdata is None:
            raise ValueError("mc_r_seqdata cannot be NULL when dissrepl is a distance method")
        from .mc_diss import mc_disslist

        dissrepl = mc_disslist(
            mc_r_seqdata,
            method=dissrepl,
            full_matrix=True,
            use_udiss=udiss,
            **distance_kwargs,
        )

    if isinstance(dissrepl, UDistResult):
        sdx = dissrepl.sdx
        n_sets = dissrepl.n_sets
        obs = dissrepl.obs
        if obs:
            diss_o = mc_extract_dist(dissrepl, k=n_sets, full_matrix=True)
            n_sets = n_sets - 1
        sum_diss = mc_extract_dist(dissrepl, k=1, full_matrix=True)
        sum_diss = _as_mat(sum_diss)
        sum_sq = sum_diss ** 2
        for k in range(2, n_sets + 1):
            dk = _as_mat(mc_extract_dist(dissrepl, k=k, full_matrix=True))
            sum_diss = sum_diss + dk
            sum_sq = sum_sq + dk ** 2
        n = n_sets
        mc_mean = sum_diss / n
        mc_sd = sum_sq / n - mc_mean ** 2
        labels = _matrix_labels(sum_diss)
    elif isinstance(dissrepl, (list, MCDissList)):
        obs = getattr(dissrepl, "obs", False)
        mats = list(dissrepl)
        if obs:
            diss_o = mats[-1]
            mats = mats[:-1]
        n = len(mats)
        labels = _matrix_labels(mats[0])
        stack = np.stack([_as_mat(m) for m in mats], axis=0)
        mc_mean = stack.mean(axis=0)
        mc_sd = stack.var(axis=0, ddof=0)
    else:
        raise TypeError("Bad dissrepl type!")

    mc_sd = np.asarray(mc_sd, dtype=float)
    mc_sd[mc_sd < 0] = 0.0
    if n and n > 1:
        mc_sd = np.sqrt(mc_sd * n / (n - 1))

    if not toref:
        mc_mean = _as_mat(mc_mean)
        mc_sd = _as_mat(mc_sd)
        if not full_matrix:
            mc_mean = squareform(mc_mean, checks=False)
            mc_sd = squareform(mc_sd, checks=False)
        else:
            mc_mean = pd.DataFrame(mc_mean, index=labels, columns=labels)
            mc_sd = pd.DataFrame(mc_sd, index=labels, columns=labels)

    ret = DistMCResult(mc_mean=mc_mean, mc_sd=mc_sd, n=n, toref=toref, obs=obs)

    if diss_o is not None:
        diss_o_mat = _as_mat(diss_o)
        mc_mean_mat = _as_mat(ret.mc_mean)
        mc_sd_mat = _as_mat(ret.mc_sd)
        bias = diss_o_mat - mc_mean_mat
        mse = mc_sd_mat ** 2 + bias ** 2
        se = np.sqrt(mse)
        if not toref and not full_matrix:
            ret.diss_o = squareform(diss_o_mat, checks=False)
            ret.mc_bias = squareform(bias, checks=False)
            ret.mc_se = squareform(se, checks=False)
            ret.mc_mse = squareform(mse, checks=False)
        else:
            ret.diss_o = pd.DataFrame(diss_o_mat, index=labels, columns=labels)
            ret.mc_bias = pd.DataFrame(bias, index=labels, columns=labels)
            ret.mc_se = pd.DataFrame(se, index=labels, columns=labels)
            ret.mc_mse = pd.DataFrame(mse, index=labels, columns=labels)
    return ret


def _as_mat(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return squareform(arr)
    return arr


def mc_ratios(
    dist_mc: DistMCResult,
    diss_o: Optional[Union[np.ndarray, pd.DataFrame]] = None,
) -> DistMCResult:
    """
    Standardized distance ratios (R ``MCratios``).

    ``diss.z = diss.o / MC.se``, ``MC.mean.z = MC.mean / mean.se`` with
    ``mean.se = MC.sd / R`` (``seqdistMCSE``) or ``MC.sd / sqrt(N)`` (``MCseqdistSE``).
    """
    if diss_o is not None:
        diss_o_mat = _as_mat(diss_o)
        mc_bias = diss_o_mat - _as_mat(dist_mc.mc_mean)
        mc_se = np.sqrt(_as_mat(dist_mc.mc_sd) ** 2 + mc_bias ** 2)
    else:
        if dist_mc.diss_o is None:
            raise ValueError("No diss.o found; provide diss_o or run MCseqdistSE with obs=TRUE.")
        diss_o_mat = _as_mat(dist_mc.diss_o)
        mc_se = _as_mat(dist_mc.mc_se) if dist_mc.mc_se is not None else _as_mat(dist_mc.mc_sd)

    r_val = dist_mc.r
    n_val = dist_mc.n
    if n_val is None:
        mean_se = _as_mat(dist_mc.mc_sd) / (r_val if r_val else 1)
    else:
        mean_se = _as_mat(dist_mc.mc_sd) / np.sqrt(n_val)

    diss_z = diss_o_mat / mc_se
    diss_z = np.where(diss_o_mat == 0, 0.0, diss_z)
    diss_z = np.minimum(diss_z, 99.0)

    mc_mean_z = _as_mat(dist_mc.mc_mean) / mean_se

    out = DistMCResult(
        mc_mean=dist_mc.mc_mean,
        mc_sd=dist_mc.mc_sd,
        n=n_val,
        r=r_val,
        diss_o=diss_o_mat,
        diss_z=diss_z,
        mc_mean_z=mc_mean_z,
        mean_se=mean_se,
        mc_se=mc_se,
        toref=dist_mc.toref,
    )
    return out
