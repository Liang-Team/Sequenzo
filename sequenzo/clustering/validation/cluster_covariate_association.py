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


def _cluster_design_matrix(cluster_labels: np.ndarray) -> np.ndarray:
    if np.all(cluster_labels == 0):
        return np.ones((len(cluster_labels), 1), dtype=np.float64)

    dummies = pd.get_dummies(pd.Series(cluster_labels), drop_first=True).astype(np.float64)
    if dummies.shape[1] == 0:
        return np.ones((len(cluster_labels), 1), dtype=np.float64)
    return np.asarray(sm.add_constant(dummies, has_constant="add"), dtype=np.float64)


def _case_weights(n_observations: int, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        case_weights = np.ones(n_observations, dtype=np.float64)
    else:
        case_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if case_weights.shape[0] != n_observations:
            raise ValueError("weights must have one value per observation.")
        if np.any(~np.isfinite(case_weights)) or np.any(case_weights < 0):
            raise ValueError("weights must be finite and non-negative.")

    if float(case_weights.sum()) <= 0:
        raise ValueError("weights must contain positive total mass.")
    return case_weights


def _linear_bic_like_r_lm(y: np.ndarray, x: np.ndarray, weights: Optional[np.ndarray]) -> float:
    case_weights = _case_weights(len(y), weights)
    positive = case_weights > 0
    y_pos = y[positive]
    x_pos = x[positive]
    w_pos = case_weights[positive]
    n_positive = len(y_pos)

    sqrt_w = np.sqrt(w_pos)
    x_weighted = x_pos * sqrt_w[:, None]
    y_weighted = y_pos * sqrt_w
    coef, _, rank, _ = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)
    residuals = y_pos - x_pos @ coef
    rss = float(np.sum(w_pos * residuals**2))

    if rss <= 0:
        return float("-inf")

    log_likelihood = 0.5 * (
        float(np.sum(np.log(w_pos)))
        - n_positive * (np.log(2.0 * np.pi) + 1.0 + np.log(rss / n_positive))
    )
    return float(-2.0 * log_likelihood + np.log(n_positive) * (int(rank) + 1))


def _categorical_bic(
    covar: np.ndarray,
    cluster_labels: np.ndarray,
    weights: Optional[np.ndarray],
) -> float:
    y = pd.Categorical(covar).codes
    if np.any(y < 0):
        raise ValueError("covar contains missing categorical values.")

    case_weights = _case_weights(len(y), weights)
    total_weight = float(case_weights.sum())

    log_likelihood = 0.0
    n_categories = int(y.max()) + 1
    for cluster in pd.unique(cluster_labels):
        mask = cluster_labels == cluster
        group_weight = float(case_weights[mask].sum())
        if group_weight <= 0:
            continue

        counts = np.bincount(y[mask], weights=case_weights[mask], minlength=n_categories)
        positive = counts > 0
        log_likelihood += float(np.sum(counts[positive] * np.log(counts[positive] / group_weight)))

    n_parameters = len(pd.unique(cluster_labels)) * (n_categories - 1)
    return float(-2.0 * log_likelihood + np.log(total_weight) * n_parameters)


def _model_bic(covar: np.ndarray, cluster_labels: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if np.issubdtype(np.asarray(covar).dtype, np.number):
        x = _cluster_design_matrix(cluster_labels)
        y = np.asarray(covar, dtype=np.float64)
        return _linear_bic_like_r_lm(y, x, weights)

    return _categorical_bic(covar, cluster_labels, weights)


def cluster_association(
    clustrange: ClusterRangeResult,
    diss: np.ndarray,
    covar: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Relate clustering solutions to a covariate (WeightedCluster ``clustassoc``).

    Numeric covariates use an R ``lm``-style BIC, while string/object/categorical
    covariates use a multinomial-style categorical BIC. Cast integer-coded
    categories to a non-numeric dtype before calling when they should be treated
    as categorical levels rather than continuous scores.

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
