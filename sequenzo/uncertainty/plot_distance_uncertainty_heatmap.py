"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_distance_uncertainty_heatmap.py
@Time    : 21/05/2026 09:15
@Desc    : Heatmap plot for distance timing uncertainty.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dist_mc_display import _as_matrix
from .mc_seqdist_se import DistMCResult

WhichMatrix = Literal[
    "diss_z",
    "mc_mean",
    "mc_sd",
    "mc_se",
    "diss_o",
    "mc_bias",
    "mean_se",
    "mc_mean_z",
]


def plot_distance_uncertainty_heatmap(
    result: DistMCResult,
    which: WhichMatrix = "diss_z",
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    figsize: tuple[float, float] = (8.0, 6.5),
    show: bool = True,
) -> plt.Axes:
    """
    Plot a heatmap of one uncertainty-related distance matrix.

    Parameters
    ----------
    result
        Output from ``get_distance_timing_uncertainty`` or
        ``get_distance_matrix_stability`` (with observed distances).
    which
        Matrix to plot: ``diss_z`` (default), ``mc_se``, ``mc_mean``, ``diss_o``,
        ``mc_bias``, ``mc_sd``, ``mean_se``, or ``mc_mean_z``.
    ax
        Optional matplotlib axes; creates a new figure if ``None``.
    show
        Call ``plt.show()`` when ``True`` and this function created the figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    field_map = {
        "diss_z": result.diss_z,
        "mc_mean": result.mc_mean,
        "mc_sd": result.mc_sd,
        "mc_se": result.mc_se,
        "diss_o": result.diss_o,
        "mc_bias": result.mc_bias,
        "mean_se": result.mean_se,
        "mc_mean_z": result.mc_mean_z,
    }
    if which not in field_map:
        raise ValueError(f"Unknown which={which!r}")
    raw = field_map[which]
    if raw is None:
        raise ValueError(
            f"Result has no {which!r} matrix. "
            "Use get_distance_timing_uncertainty(..., ratios=True) or include observed distances."
        )

    if isinstance(raw, pd.DataFrame):
        mat = raw.to_numpy(dtype=float)
        labels = list(raw.index.astype(str))
    else:
        mat = _as_matrix(raw)
        labels = [str(i + 1) for i in range(mat.shape[0])]

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    n = mat.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title or f"Distance uncertainty: {which}")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label or which)

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return ax
