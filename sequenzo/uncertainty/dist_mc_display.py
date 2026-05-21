"""
@Author  : Yuqi Liang 梁彧祺
@File    : dist_mc_display.py
@Time    : 13/05/2026 17:05
@Desc    : Print and summary formatters for DistMCResult.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from .mc_seqdist_se import DistMCResult
from sequenzo.utils.weighted_stats import weighted_five_number_summary

WhatKind = Literal["all", "both", "mean", "sd", "obs", "se", "bias"]


def _as_matrix(x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return squareform(arr, checks=False)
    return arr


def _labels_from(result: DistMCResult, mat: Union[np.ndarray, pd.DataFrame]) -> list:
    if isinstance(mat, pd.DataFrame):
        return list(mat.index.astype(str))
    n = _as_matrix(mat).shape[0]
    return [str(i + 1) for i in range(n)]


def _pairwise_weights(n: int, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weights for lower-triangle entries (R ``tcrossprod(weights)``)."""
    if weights is None or len(weights) == 1:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
    return squareform(np.outer(w, w), checks=False)


def _subset_matrix(
    mat: np.ndarray,
    labels: list,
    n: int,
) -> Union[np.ndarray, pd.DataFrame]:
    """First ``n`` sequences (0 = all)."""
    if n <= 0 or n >= mat.shape[0]:
        return mat
    sub = mat[:n, :n]
    lab = labels[:n]
    return pd.DataFrame(sub, index=lab, columns=lab)


def _print_block(title: str, mat: Union[np.ndarray, pd.DataFrame], n: int, labels: list) -> None:
    print(f"\n{title}")
    view = _subset_matrix(_as_matrix(mat), labels, n)
    with pd.option_context("display.max_columns", 20, "display.width", 120, "display.precision", 4):
        print(view)


def print_dist_mc(
    result: DistMCResult,
    n: int = 6,
    what: WhatKind = "all",
) -> DistMCResult:
    """
    Print key distance-uncertainty tables (R ``print.distMC``).

    Parameters
    ----------
    result
        Output from ``seqdist_mcse`` or ``mc_seqdist_se``.
    n
        Number of sequences to show (0 = all). Default 6.
    what
        ``"all"``, ``"both"`` (mean + sd), ``"mean"``, ``"sd"``, ``"obs"``,
        ``"se"``, or ``"bias"``.
    """
    if what not in ("all", "both", "mean", "sd", "obs", "se", "bias"):
        raise ValueError('what must be one of "all", "both", "mean", "sd", "obs", "se", "bias"')

    base = result.mc_mean if result.mc_mean is not None else result.mc_sd
    labels = _labels_from(result, base)
    ns = len(labels)
    r_val = result.r
    n_rep = (r_val * r_val) if r_val else result.n
    rep_txt = (
        f"{ns} sequences, R = {r_val}, {n_rep} MC-simulated dissimilarities per observed pair"
        if r_val
        else f"{ns} sequences, N = {result.n} MC-replicated distance matrices"
    )
    print(rep_txt)

    if result.diss_o is not None and what in ("obs", "all"):
        _print_block("diss.o: Observed dissimilarities", result.diss_o, n, labels)
    if what in ("mean", "both", "all"):
        _print_block("MC.mean: Mean of MC-simulated dissimilarities", result.mc_mean, n, labels)
    if result.mc_sd is not None and what in ("sd", "both", "all"):
        _print_block("MC.sd: Standard deviation of MC-simulated dissimilarities", result.mc_sd, n, labels)
    if result.mc_se is not None and what in ("se", "all"):
        _print_block("MC.se: Standard error of dissimilarities", result.mc_se, n, labels)
    if result.diss_z is not None and what == "all":
        _print_block("diss.z: Ratios diss.o / MC.se", result.diss_z, n, labels)
    if result.mc_mean_z is not None and what == "all":
        _print_block("MC.mean.z: Ratios MC.mean / mean.se", result.mc_mean_z, n, labels)
    if result.mean_se is not None and what == "all":
        _print_block("mean.se: Standard error of mean simulated dissimilarities", result.mean_se, n, labels)
    if result.mc_bias is not None and what in ("bias", "all"):
        _print_block("MC.bias: Observed minus MC.mean", result.mc_bias, n, labels)
    return result


def summary_dist_mc(
    result: DistMCResult,
    *,
    weights: Optional[np.ndarray] = None,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Weighted five-number summaries over all pairwise distances (R ``summary.distMC``).

    Returns a DataFrame with rows Min, Q1, Median, Q3, Max and columns for each
    statistic that is available in ``result``.
    """
    mat = _as_matrix(result.mc_mean)
    n = mat.shape[0]
    w_pair = _pairwise_weights(n, weights if weights is not None else result.weights)
    lower = mat[np.triu_indices(n, k=1)]

    if not np.any(lower > 0) or n <= 1 or np.isnan(np.sum(w_pair)):
        if not silent:
            print("[!] summary not applicable (empty or single sequence).")
        return pd.DataFrame()

    row_names = ["Min", "Q1", "Median", "Q3", "Max"]
    cols = {}

    def _add_col(name: str, obj: Optional[Union[np.ndarray, pd.DataFrame]]) -> None:
        if obj is None:
            return
        m = _as_matrix(obj)
        vals = m[np.triu_indices(n, k=1)]
        cols[name] = weighted_five_number_summary(vals, weights=w_pair)

    if result.diss_o is not None:
        _add_col("diss", result.diss_o)
    _add_col("MC.mean", result.mc_mean)
    _add_col("MC.sd", result.mc_sd)
    if result.mean_se is not None:
        _add_col("mean.se", result.mean_se)
    if result.mc_mean_z is not None:
        _add_col("MC.mean/mean.se", result.mc_mean_z)
    if result.mc_se is not None:
        _add_col("MC.se", result.mc_se)
    if result.mc_bias is not None:
        _add_col("MC.bias", result.mc_bias)

    if not cols:
        return pd.DataFrame()

    frame = pd.DataFrame(cols, index=row_names)
    if not silent:
        r_val = result.r
        if r_val:
            print(
                f"{n} sequences, R = {r_val}, {r_val * r_val} "
                "MC-simulated dissimilarities per observed pair"
            )
        elif result.n:
            print(f"{n} sequences, N = {result.n} MC-replicated distance matrices")
    return frame


def format_dist_mc_brief(result: DistMCResult) -> str:
    """One-line overview for notebooks and ``repr``."""
    mat = _as_matrix(result.mc_mean)
    n = mat.shape[0]
    r_val = result.r
    if r_val:
        return (
            f"DistMCResult({n} sequences, R={r_val}, "
            f"{r_val * r_val} MC draws per pair; call print_dist_mc() for tables)"
        )
    return f"DistMCResult({n} sequences, N={result.n} MC sets; call print_dist_mc() for tables)"


def _dist_mc_print(self, n: int = 6, what: WhatKind = "all") -> DistMCResult:
    return print_dist_mc(self, n=n, what=what)


def _dist_mc_summary(self, weights=None, silent: bool = False) -> pd.DataFrame:
    return summary_dist_mc(self, weights=weights, silent=silent)


DistMCResult.__repr__ = lambda self: format_dist_mc_brief(self)  # type: ignore[method-assign]
DistMCResult.print = _dist_mc_print  # type: ignore[method-assign]
DistMCResult.summary = _dist_mc_summary  # type: ignore[method-assign]
