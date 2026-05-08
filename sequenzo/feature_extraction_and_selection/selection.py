"""
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
        random_state=random_state,
        verbose=verbose,
    )

    selected_mask = boruta_result.selected_mask
    selected_feature_names = [
        feature_names[i] for i, keep in enumerate(selected_mask) if keep
    ]
    return {
        "selected_mask": selected_mask,
        "selected_indices": boruta_result.selected_indices,
        "selected_feature_names": selected_feature_names,
        "hit_counts": boruta_result.hit_counts,
        "shadow_hit_counts": boruta_result.shadow_hit_counts,
    }


__all__ = ["select_relevant_features"]
