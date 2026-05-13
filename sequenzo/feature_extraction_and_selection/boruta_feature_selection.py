"""
@Author  : Yuqi Liang 梁彧祺
@File    : boruta_feature_selection.py
@Time    : 21/03/2026 11:30
@Desc    :
    Boruta-based all-relevant feature selection via BorutaPy (PyPI package ``boruta``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from boruta import BorutaPy
except ImportError as exc:
    raise ImportError(
        "The 'boruta' package is required for feature selection in Sequenzo. "
        "It should be installed automatically with sequenzo (pip install sequenzo). "
        "If missing, run: pip install boruta"
    ) from exc


@dataclass(frozen=True)
class BorutaFeatureSelectionResult:
    selected_mask: np.ndarray
    selected_indices: List[int]
    tentative_mask: Optional[np.ndarray] = None
    tentative_indices: Optional[List[int]] = None
    ranking: Optional[np.ndarray] = None
    hit_counts: Optional[np.ndarray] = None
    shadow_hit_counts: Optional[np.ndarray] = None


def select_all_relevant_features_boruta(
    X: np.ndarray,
    y: np.ndarray,
    *,
    problem_type: str,
    estimator: Optional[BaseEstimator] = None,
    n_iter: int = 50,
    perc: float = 100.0,
    random_state: Optional[int] = None,
    verbose: bool = False,
    boruta_max_iter: Optional[int] = None,
    boruta_alpha: float = 0.01,
    boruta_two_step: bool = False,
) -> BorutaFeatureSelectionResult:
    """
    All-relevant feature selection using BorutaPy.

    Requires the ``boruta`` package (declared as a Sequenzo runtime dependency).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be 'regression' or 'classification'.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if problem_type == "regression":
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")
    elif np.any(y < 0):
        raise ValueError("y contains invalid category codes.")

    if estimator is None:
        if problem_type == "regression":
            estimator = RandomForestRegressor(
                n_estimators=800, n_jobs=-1, random_state=random_state, max_features="sqrt"
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=800, n_jobs=-1, random_state=random_state, max_features="sqrt"
            )

    n_iter_external = boruta_max_iter if boruta_max_iter is not None else n_iter
    boruta = BorutaPy(
        clone(estimator),
        n_estimators="auto",
        max_iter=n_iter_external,
        alpha=boruta_alpha,
        perc=perc,
        two_step=boruta_two_step,
        random_state=random_state,
    )
    boruta.fit(X, y)
    selected_mask = boruta.support_
    selected_indices = np.where(selected_mask)[0].tolist()
    tentative_mask = boruta.support_weak_
    tentative_indices = np.where(tentative_mask)[0].tolist()
    return BorutaFeatureSelectionResult(
        selected_mask=selected_mask,
        selected_indices=selected_indices,
        tentative_mask=tentative_mask,
        tentative_indices=tentative_indices,
        ranking=np.asarray(boruta.ranking_, dtype=int),
    )
