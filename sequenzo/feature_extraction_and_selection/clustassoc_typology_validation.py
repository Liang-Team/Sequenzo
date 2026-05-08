"""
@Author  : Yuqi Liang 梁彧祺
@File    : clustassoc_typology_validation.py
@Time    : 22/03/2026 16:14
@Desc    :
    Clustassoc-like typology validation for sequence clustering solutions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.discrepancy_analysis.dissassoc_permutation import distance_multifactor_anova


def _one_hot(x: np.ndarray, *, drop_first: bool = False) -> np.ndarray:
    s = pd.Series(np.asarray(x))
    return pd.get_dummies(s, drop_first=drop_first).to_numpy(dtype=float)


def _compute_pseudo_r2_for_terms(
    *,
    diss: np.ndarray,
    design: np.ndarray,
    term_ids: Sequence[int],
    term_labels: Sequence[str],
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    res = distance_multifactor_anova(
        distance_matrix=np.asarray(diss, dtype=float),
        design_matrix=np.asarray(design, dtype=float),
        term_ids=np.asarray(term_ids, dtype=int),
        term_labels=list(term_labels),
        weights=weights,
        gower=False,
        squared=False,
        R=0,
    )
    return res["summary"]


def clustassoc_like_typology_validation(
    diss: np.ndarray,
    covariate: Union[np.ndarray, Sequence[Any]],
    clustering_labels_by_k: Dict[int, np.ndarray],
    *,
    sample_weights: Optional[np.ndarray] = None,
    covariate_is_categorical: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    D = np.asarray(diss, dtype=float)
    y = np.asarray(covariate)
    n = D.shape[0]

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("diss must be a square matrix.")
    if y.shape[0] != n:
        raise ValueError("covariate length must match diss matrix size.")

    if sample_weights is None:
        w = None
    else:
        w = np.asarray(sample_weights, dtype=float)
        if w.ndim != 1 or len(w) != n:
            raise ValueError("sample_weights must be a 1D array with length n.")

    if covariate_is_categorical:
        X_cov = _one_hot(y, drop_first=False)
        design_null = X_cov
        term_ids_null = [0] * X_cov.shape[1]
    else:
        design_null = y.reshape(-1, 1).astype(float)
        term_ids_null = [0]

    summary_null = _compute_pseudo_r2_for_terms(
        diss=D,
        design=design_null,
        term_ids=term_ids_null,
        term_labels=["Covariate"],
        weights=w,
    )
    pseudo_r2_null = float(
        summary_null.loc[summary_null["Variable"] == "Covariate", "PseudoR2"].iloc[0]
    )

    rows = []
    for k, labels in sorted(clustering_labels_by_k.items(), key=lambda kv: kv[0]):
        lab = np.asarray(labels)
        if lab.shape[0] != n:
            raise ValueError(f"Length mismatch for clustering_labels_by_k[{k}].")

        X_clust = _one_hot(lab, drop_first=False)
        if covariate_is_categorical:
            X_cov = _one_hot(y, drop_first=False)
            design_full = np.hstack([X_clust, X_cov])
            term_ids_full = [0] * X_clust.shape[1] + [1] * X_cov.shape[1]
        else:
            X_cov = y.reshape(-1, 1).astype(float)
            design_full = np.hstack([X_clust, X_cov])
            term_ids_full = [0] * X_clust.shape[1] + [1]

        if verbose:
            print(f"[clustassoc-like] evaluating k={k}")

        summary_full = _compute_pseudo_r2_for_terms(
            diss=D,
            design=design_full,
            term_ids=term_ids_full,
            term_labels=["Clustering", "Covariate"],
            weights=w,
        )
        remaining = float(
            summary_full.loc[summary_full["Variable"] == "Covariate", "PseudoR2"].iloc[0]
        )
        unaccounted = remaining / pseudo_r2_null if pseudo_r2_null > 0 else np.nan
        accounted = 1.0 - unaccounted if np.isfinite(unaccounted) else np.nan

        rows.append(
            {
                "k": int(k),
                "pseudoR2_null": pseudo_r2_null,
                "pseudoR2_remaining": remaining,
                "unaccounted_share": unaccounted,
                "accounted_share": accounted,
            }
        )

    return pd.DataFrame(rows)

