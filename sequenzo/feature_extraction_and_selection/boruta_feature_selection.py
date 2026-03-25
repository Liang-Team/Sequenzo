"""
@Author  : Yuqi Liang 梁彧祺
@File    : boruta_feature_selection.py
@Time    : 21/03/2026 11:30
@Desc    :
    Boruta-based all-relevant feature selection with external BorutaPy support
    and an internal Boruta-like fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass(frozen=True)
class BorutaFeatureSelectionResult:
    selected_mask: np.ndarray
    selected_indices: List[int]
    hit_counts: Optional[np.ndarray] = None
    shadow_hit_counts: Optional[np.ndarray] = None


def _boruta_like_select(
    X: np.ndarray,
    y: np.ndarray,
    *,
    estimator: BaseEstimator,
    n_iter: int = 50,
    perc: float = 100.0,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> BorutaFeatureSelectionResult:
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y must have the same number of rows.")

    hit_counts = np.zeros(p, dtype=int)
    shadow_hit_counts = np.zeros(p, dtype=int)

    for it in range(n_iter):
        X_shadow = np.empty_like(X)
        for j in range(p):
            X_shadow[:, j] = rng.permutation(X[:, j])

        X_aug = np.hstack([X, X_shadow])
        est = clone(estimator)
        if hasattr(est, "random_state"):
            try:
                est.set_params(random_state=int(rng.integers(0, 1_000_000_000)))
            except Exception:
                pass
        est.fit(X_aug, y)

        if not hasattr(est, "feature_importances_"):
            raise ValueError("Estimator must expose feature_importances_.")
        imps = np.asarray(est.feature_importances_, dtype=float)
        if imps.shape[0] != 2 * p:
            raise ValueError("Unexpected feature_importances_ length from estimator.")

        real_imp = imps[:p]
        shadow_imp = imps[p:]
        threshold = float(np.percentile(shadow_imp, perc))
        hit_counts += (real_imp > threshold).astype(int)
        shadow_hit_counts += (shadow_imp > threshold).astype(int)

        if verbose and (it % max(1, n_iter // 10) == 0):
            print(f"[Boruta-like] iter {it+1}/{n_iter}, threshold={threshold:.6g}")

    shadow_max_hit = int(np.max(shadow_hit_counts)) if shadow_hit_counts.size else 0
    selected_mask = hit_counts > shadow_max_hit
    selected_indices = np.where(selected_mask)[0].tolist()
    return BorutaFeatureSelectionResult(
        selected_mask=selected_mask,
        selected_indices=selected_indices,
        hit_counts=hit_counts,
        shadow_hit_counts=shadow_hit_counts,
    )


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
    boruta_alpha: float = 0.05,
    try_external_first: bool = True,
) -> BorutaFeatureSelectionResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be 'regression' or 'classification'.")

    if estimator is None:
        if problem_type == "regression":
            estimator = RandomForestRegressor(
                n_estimators=800, n_jobs=-1, random_state=random_state, max_features="sqrt"
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=800, n_jobs=-1, random_state=random_state, max_features="sqrt"
            )

    if try_external_first:
        try:
            from boruta import BorutaPy  # type: ignore

            n_iter_external = boruta_max_iter if boruta_max_iter is not None else n_iter
            boruta = BorutaPy(
                clone(estimator),
                n_estimators="auto",
                max_iter=n_iter_external,
                alpha=boruta_alpha,
                perc=perc,
                random_state=random_state,
            )
            boruta.fit(X, y)
            selected_mask = boruta.support_
            selected_indices = np.where(selected_mask)[0].tolist()
            return BorutaFeatureSelectionResult(
                selected_mask=selected_mask,
                selected_indices=selected_indices,
            )
        except ModuleNotFoundError:
            if verbose:
                print("[Boruta] External package not found, using fallback.")
        except Exception as e:
            if verbose:
                print(f"[Boruta] External BorutaPy failed ({e!r}), using fallback.")

    return _boruta_like_select(
        X,
        y,
        estimator=estimator,
        n_iter=n_iter,
        perc=perc,
        random_state=random_state,
        verbose=verbose,
    )

