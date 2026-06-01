"""
@Author  : Yuqi Liang 梁彧祺
@File    : results.py
@Time    : 19/05/2026 19:33
@Desc    : 
Result containers for multidomain CLARA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MDClaraResult:
    """
    Output of :func:`md_clara`.

    ``stats`` columns include ``total_diss`` (weighted total nearest-medoid distance)
    and ``avg_dist`` (weighted average distance). Repetition selection for
    ``criteria='distance'`` uses ``total_diss``.
    """

    strategy: str
    method: str
    kvals: List[int]
    best_by_k: Dict[int, Dict[str, Any]]
    clustering: pd.DataFrame
    stats: pd.DataFrame
    medoids: Dict[int, np.ndarray]
    settings: Dict[str, Any]
    stability: Optional[Dict[int, Dict[str, Any]]] = None
    membership: Optional[Dict[int, np.ndarray]] = field(default=None)
    combined_state_space: Optional[Dict[str, Any]] = None
    subsample_diagnostics: Optional[pd.DataFrame] = None
    route_diagnostics: Optional[Dict[str, Any]] = None

    def best_clustering(self, k: int) -> np.ndarray:
        """Return one-based integer cluster labels for the requested k."""
        column = f"Cluster {k}"
        if column not in self.clustering.columns:
            raise KeyError(f"No clustering stored for k={k}.")
        return self.clustering[column].to_numpy(dtype=int, copy=True)


__all__ = ["MDClaraResult"]
