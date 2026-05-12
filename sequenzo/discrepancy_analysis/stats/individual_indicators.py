"""Individual discrepancy indicators (TraMineRextras dissindic)."""

import numpy as np
import pandas as pd
from typing import Optional

from .multifactor_association import gower_matrix, _weighted_hat_matrix_qr

def individual_indicators(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    weights: Optional[np.ndarray] = None,
    gower: bool = False,
    squared: bool = False,
) -> pd.DataFrame:
    """Individual marginality and gain indicators (dissindic)."""
    D = np.asarray(distance_matrix, dtype=float)
    g = np.asarray(group)
    n = D.shape[0]

    if D.shape[0] != D.shape[1]:
        raise ValueError("[individual_indicators] distance_matrix must be square.")
    if g.shape[0] != n:
        raise ValueError("[individual_indicators] group length must match n.")

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    group_series = pd.Series(g)
    X = pd.get_dummies(group_series, drop_first=False).to_numpy(dtype=float)
    G = D if gower else gower_matrix(D, squared=squared, weights=weights)

    w_mat = np.zeros_like(G)
    np.fill_diagonal(w_mat, weights)

    residuals = np.zeros((n, 2), dtype=float)
    residuals[:, 0] = np.diag(G)

    hat = _weighted_hat_matrix_qr(X, weights=weights)
    resid_mat = (w_mat - hat) @ G
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals[:, 1] = np.diag(resid_mat) / weights

    marginality = residuals[:, 1]
    gain = residuals[:, 0] - residuals[:, 1]
    return pd.DataFrame({"group": g, "marginality": marginality, "gain": gain})
