"""
@Author  : Yuqi Liang 梁彧祺
@File    : cluster_covariate_association.py
@Time    : 07/05/2025 17:51
@Desc    :
Clustering versus covariate association (WeightedCluster ``clustassoc``).
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sequenzo.discrepancy_analysis.stats.multifactor_association import multifactor_association

from .partition_quality import ClusterRangeResult

ClustassocStat = Literal["Unaccounted", "Remaining", "BIC"]

_STAT_YLABELS = {
    "Unaccounted": "Unreproduced proportion of the association",
    "Remaining": "Unexplained association",
    "BIC": "BIC",
}


def _model_bic(covar: np.ndarray, cluster_labels: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if np.issubdtype(np.asarray(covar).dtype, np.number):
        y = np.asarray(covar, dtype=np.float64)
        if np.all(cluster_labels == 0):
            x = np.ones((len(y), 1), dtype=np.float64)
        else:
            x = sm.add_constant(
                pd.get_dummies(pd.Series(cluster_labels), drop_first=False).astype(np.float64)
            )
        if weights is None:
            model = sm.OLS(y, x).fit()
        else:
            model = sm.WLS(y, x, weights=weights).fit()
        return float(model.bic)

    y = pd.Categorical(covar)
    if np.all(cluster_labels == 0):
        x = np.ones((len(y), 1), dtype=np.float64)
        model = sm.MNLogit(y.codes, x).fit(disp=0, weights=weights)
        return float(model.bic)
    x = sm.add_constant(pd.get_dummies(pd.Series(cluster_labels), drop_first=False).astype(np.float64))
    model = sm.MNLogit(y.codes, x).fit(disp=0, weights=weights)
    return float(model.bic)


def cluster_association(
    clustrange: ClusterRangeResult,
    diss: np.ndarray,
    covar: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Relate clustering solutions to a covariate (WeightedCluster ``clustassoc``).

    Returns a table with columns ``Unaccounted``, ``Remaining``, ``BIC``, and
    ``numcluster``, plus a baseline row ``No Clustering``.
    """
    diss = np.asarray(diss, dtype=np.float64)
    covar = np.asarray(covar).reshape(-1)
    if diss.shape[0] != covar.shape[0]:
        raise ValueError("diss and covar must refer to the same observations.")

    null_mm = multifactor_association(
        diss, pd.DataFrame({"covar": covar}), weights=weights
    )
    null_bic = _model_bic(covar, np.zeros_like(covar), weights)
    null_covar = float(null_mm.loc["covar", "PseudoR2"])

    rows = []
    for idx, column in enumerate(clustrange.clustering.columns):
        cluster_labels = clustrange.clustering.iloc[:, idx].to_numpy()
        mm = multifactor_association(
            diss,
            pd.DataFrame(
                {
                    "cluster": pd.Categorical(cluster_labels),
                    "covar": covar,
                }
            ),
            weights=weights,
        )
        cluster_r2 = float(mm.loc["cluster", "PseudoR2"])
        covar_r2 = float(mm.loc["covar", "PseudoR2"])
        total_r2 = float(mm.loc["Total", "PseudoR2"])
        denom = total_r2 - cluster_r2
        rows.append(
            {
                "Unaccounted": covar_r2 / denom if denom else np.nan,
                "Remaining": covar_r2,
                "BIC": _model_bic(covar, cluster_labels, weights),
                "numcluster": int(clustrange.kvals[idx]),
            }
        )

    result = pd.DataFrame(rows, index=clustrange.clustering.columns)
    baseline = pd.DataFrame(
        {
            "Unaccounted": [1.0],
            "Remaining": [null_covar],
            "BIC": [null_bic],
            "numcluster": [1],
        },
        index=["No Clustering"],
    )
    out = pd.concat([baseline, result])
    out.attrs["clustassoc_class"] = True
    return out


def plot_cluster_association(
    clustassoc_table: pd.DataFrame,
    *,
    stat: Union[ClustassocStat, str] = "Unaccounted",
    plot_type: str = "b",
    title: Optional[str] = None,
    xlabel: str = "Number of clusters",
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    save_as: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
    **plot_kwargs,
) -> plt.Axes:
    """
    Plot ``clustassoc`` diagnostics (WeightedCluster ``plot.clustassoc``).

    Parameters
    ----------
    clustassoc_table
        Output of :func:`cluster_association`.
    stat
        Column to plot: ``Unaccounted`` (default), ``Remaining``, or ``BIC``.
    plot_type
        Matplotlib plot type (default ``"b"``: markers and lines).
    """
    if stat not in _STAT_YLABELS:
        raise ValueError(f"stat must be one of {list(_STAT_YLABELS)}.")

    required = {"numcluster", stat}
    missing = required - set(clustassoc_table.columns)
    if missing:
        raise ValueError(f"clustassoc_table is missing columns: {sorted(missing)}")

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=figsize)

    x = clustassoc_table["numcluster"].to_numpy(dtype=float)
    y = clustassoc_table[stat].to_numpy(dtype=float)
    ax.plot(x, y, plot_type, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or _STAT_YLABELS[stat])
    if title is not None:
        ax.set_title(title)

    if created_fig:
        fig = ax.figure
        fig.tight_layout()
        if save_as:
            fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        elif save_as:
            plt.close(fig)

    return ax
