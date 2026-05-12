"""Marginal single-factor association per covariate column (TraMineR: repeated dissassoc)."""

import numpy as np
import pandas as pd
from typing import Optional

from .single_factor_association import single_factor_association


def marginal_factor_association(
    distance_matrix: np.ndarray,
    factors: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    R: int = 0,
    weight_permutation: Optional[str] = None,
    squared: bool = False
) -> pd.DataFrame:
    """
    Marginal single-factor dissassoc() summaries for each covariate column.

    This helper runs one single_factor_association() call per column. It is
    not equivalent to TraMineR's multifactor_association(), which fits a multifactor model.

    Parameters
    ----------
    distance_matrix : np.ndarray or pandas.DataFrame
        Square symmetric distance matrix of shape (n, n).
    factors : pandas.DataFrame
        DataFrame with n rows and one or more factor columns describing
        covariates (e.g., gender, cohort, country). Each column is treated
        as a separate factor in the analysis.
    weights : np.ndarray, optional
        Optional weights for each observation (length n). If None, equal
        weights are used.
    R : int, default 0
        Number of permutations for each factor-specific association test.
        When R=0, permutation p-values are skipped (faster).
    weight_permutation : {"none", "replicate", "diss", "group"}, optional
        Weight handling strategy for permutation tests. Default: None (resolved to
        "none" without weights, otherwise "replicate").
    squared : bool, default False
        Whether to square distances before analysis, passed to
        `single_factor_association`.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per factor and the following columns:
            - 'factor'     : factor name (column name in `factors`)
            - 'Pseudo R2'  : proportion of variance explained
            - 'Pseudo F'   : pseudo F statistic
            - 'p-value'    : permutation p-value for Pseudo F (if R > 1)
            - 'n_groups'   : number of non-empty groups for that factor
    """
    # Convert distance matrix to ndarray if needed
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values

    results = []
    for col in factors.columns:
        factor_values = factors[col].values
        assoc = single_factor_association(
            distance_matrix=distance_matrix,
            group=factor_values,
            weights=weights,
            R=R,
            weight_permutation=weight_permutation,
            squared=squared,
        )

        # Count non-empty groups for information
        n_groups = (assoc["groups"]["n"] > 0).sum() - 1  # subtract "Total" row

        results.append(
            {
                "factor": col,
                "Pseudo R2": assoc["pseudo_r2"],
                "Pseudo F": assoc["pseudo_f"],
                "p-value": assoc["pseudo_f_pval"],
                "n_groups": int(n_groups),
            }
        )

    return pd.DataFrame(results).set_index("factor")
