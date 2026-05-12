"""
@Author  : Yuqi Liang 梁彧祺
@File    : cluster_covariate_association.py
@Time    : 07/05/2025 17:51
@Desc    : 
Clustering versus covariate association (WeightedCluster ``clustassoc``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .dissmfacw_factors import dissmfacw_table
from .partition_quality import ClusterRangeResult


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
    """
    diss = np.asarray(diss, dtype=np.float64)
    covar = np.asarray(covar).reshape(-1)
    if diss.shape[0] != covar.shape[0]:
        raise ValueError("diss and covar must refer to the same observations.")

    covar_frame = pd.DataFrame({"covar": covar})
    null_table = dissmfacw_table(diss, covar_frame, weights=weights)
    null_bic = _model_bic(covar, np.zeros_like(covar), weights)

    rows = []
    for idx, column in enumerate(clustrange.clustering.columns):
        cluster_labels = clustrange.clustering.iloc[:, idx].to_numpy()
        factors = pd.DataFrame({"cluster": cluster_labels, "covar": covar})
        mm = dissmfacw_table(diss, factors, weights=weights)
        remaining = float(mm.loc[mm["Variable"] == "covar", "PseudoR2"].iloc[0])
        total = float(mm.loc[mm["Variable"] == "Total", "PseudoR2"].iloc[0])
        null_covar = float(null_table.loc[null_table["Variable"] == "covar", "PseudoR2"].iloc[0])
        rows.append(
            {
                "Unaccounted": remaining / (total - null_covar) if total != null_covar else np.nan,
                "Remaining": remaining,
                "BIC": _model_bic(covar, cluster_labels, weights),
                "numcluster": int(clustrange.kvals[idx]),
            }
        )

    result = pd.DataFrame(rows, index=clustrange.clustering.columns)
    baseline = pd.DataFrame(
        {
            "Unaccounted": [1.0],
            "Remaining": [float(null_table.loc[null_table["Variable"] == "covar", "PseudoR2"].iloc[0])],
            "BIC": [null_bic],
            "numcluster": [1],
        },
        index=["No Clustering"],
    )
    return pd.concat([baseline, result])
