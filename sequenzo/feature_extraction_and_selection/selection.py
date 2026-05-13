"""
@Author  : Yuqi Liang 梁彧祺
@File    : selection.py
@Time    : 19/03/2026 07:20
@Desc    :
@Desc: Feature selection entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .boruta_feature_selection import select_all_relevant_features_boruta


def select_relevant_features(
    X: Any,
    y: Any,
    *,
    problem_type: str = "regression",
    n_iter: int = 50,
    perc: float = 100.0,
    boruta_alpha: float = 0.01,
    boruta_two_step: bool = False,
    random_state: Optional[int] = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"X{i+1}" for i in range(X_arr.shape[1])]

    boruta_result = select_all_relevant_features_boruta(
        X_arr,
        y_arr,
        problem_type=problem_type,
        n_iter=n_iter,
        perc=perc,
        boruta_alpha=boruta_alpha,
        boruta_two_step=boruta_two_step,
        random_state=random_state,
        verbose=verbose,
    )

    selected_mask = boruta_result.selected_mask
    selected_feature_names = [
        feature_names[i] for i, keep in enumerate(selected_mask) if keep
    ]
    tentative_feature_names = []
    if boruta_result.tentative_mask is not None:
        tentative_feature_names = [
            feature_names[i]
            for i, keep in enumerate(boruta_result.tentative_mask)
            if keep
        ]
    return {
        "selected_mask": selected_mask,
        "selected_indices": boruta_result.selected_indices,
        "selected_feature_names": selected_feature_names,
        "tentative_mask": boruta_result.tentative_mask,
        "tentative_indices": boruta_result.tentative_indices,
        "tentative_feature_names": tentative_feature_names,
        "boruta_ranking": boruta_result.ranking,
        "hit_counts": boruta_result.hit_counts,
        "shadow_hit_counts": boruta_result.shadow_hit_counts,
    }


__all__ = ["select_relevant_features"]
