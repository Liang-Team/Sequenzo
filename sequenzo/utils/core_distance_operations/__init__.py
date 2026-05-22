"""
Core distance operations shared across Sequenzo modules.

This package provides low-level, performance-critical primitives for
distance-matrix-based methods. Implementations are backed by the compiled
C/C++ extension and are intentionally agnostic to any specific high-level
method (clustering, discrepancy analysis, etc.).
"""

from __future__ import annotations

import importlib
import warnings
from typing import Optional

import numpy as np

_c_extension: Optional[object] = None
_c_extension_unavailable = False


def _get_c_extension() -> Optional[object]:
    """Load the compiled core extension used by distance primitives."""
    global _c_extension, _c_extension_unavailable
    if _c_extension is not None:
        return _c_extension
    if _c_extension_unavailable:
        return None
    try:
        _c_extension = importlib.import_module(
            "sequenzo.utils.core_distance_operations.core_distance_c_code"
        )
    except Exception as exc:
        _c_extension_unavailable = True
        warnings.warn(
            "C extension 'sequenzo.utils.core_distance_operations.core_distance_c_code' "
            f"could not be loaded ({exc}); using Python fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return _c_extension


def _weighted_inertia_contrib_python(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Pure-Python TraMineR-style weighted inertia contribution."""
    dist = np.asarray(distance_matrix, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int32)
    w = np.asarray(weights, dtype=np.float64)

    group_weights = w[idx]
    total_weight = float(group_weights.sum())
    ilen = len(idx)
    if total_weight <= 0 or ilen == 0:
        return np.zeros(ilen, dtype=np.float64)

    sub_dist = dist[np.ix_(idx, idx)]
    result = np.zeros(ilen, dtype=np.float64)
    for i in range(ilen):
        for j in range(i + 1, ilen):
            diss = sub_dist[i, j]
            result[i] += diss * group_weights[j]
            result[j] += diss * group_weights[i]
    return result / total_weight


def weighted_inertia_contrib(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    TraMineR-style weighted inertia contribution on selected indices.

    For each i in `indices`, returns:
        sum_j(w_j * d_ij) / sum_j(w_j)
    where j also ranges over `indices`.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)

    cmod = _get_c_extension()
    if cmod is not None:
        engine = cmod.weightedinertia(distance_matrix, indices, weights)
        return np.asarray(engine.tmrWeightedInertiaContrib(), dtype=np.float64)
    return _weighted_inertia_contrib_python(distance_matrix, indices, weights)


__all__ = ["weighted_inertia_contrib"]

