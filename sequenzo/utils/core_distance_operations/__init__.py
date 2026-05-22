"""
Core distance operations shared across Sequenzo modules.

This package provides low-level, performance-critical primitives for
distance-matrix-based methods. Implementations are backed by the compiled
C/C++ extension and are intentionally agnostic to any specific high-level
method (clustering, discrepancy analysis, etc.).
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np

_c_extension: Optional[object] = None


def _get_c_extension():
    """Load the compiled core extension used by distance primitives."""
    global _c_extension
    if _c_extension is None:
        try:
            _c_extension = importlib.import_module(
                "sequenzo.utils.core_distance_operations.core_distance_c_code"
            )
        except Exception as exc:
            raise ImportError(
                "C extension 'sequenzo.utils.core_distance_operations.core_distance_c_code' is required "
                "for core distance operations."
            ) from exc
    return _c_extension


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
    cmod = _get_c_extension()
    engine = cmod.weightedinertia(
        np.asarray(distance_matrix, dtype=np.float64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )
    return np.asarray(engine.tmrWeightedInertiaContrib(), dtype=np.float64)


__all__ = ["weighted_inertia_contrib"]
