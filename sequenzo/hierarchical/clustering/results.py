"""
@Author  : 梁彧祺 Yuqi Liang
@File    : results.py
@Time    : 28/04/2026 22:18
@Desc    :
    Result objects for pair-level trajectory typology and level clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class HierarchicalClusterResult:
    """Cluster labels at pair, level-1, or level-2 resolution (full distance matrix)."""

    level: str
    cluster_labels: np.ndarray
    unit_ids: np.ndarray
    k: int
    medoid_indices: np.ndarray
    distance_matrix: np.ndarray
    method: str = "PAMonce"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unit_id": self.unit_ids,
                "cluster": self.cluster_labels,
            }
        )


@dataclass
class PairTypologyResult:
    """
    Scalable typology of pair-level relational trajectories.

    Produced by PAM or CLARA-style representative clustering. The class name
    describes the analytical output, not the computational algorithm.
    """

    level: str
    k: int
    cluster_labels: np.ndarray
    unit_ids: np.ndarray
    medoid_indices: np.ndarray
    medoid_ids: np.ndarray
    distance_to_medoids: np.ndarray
    method: str = "CLARA"
    quality: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)
    membership: Optional[np.ndarray] = None
    representativeness: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    level_1_ids: Optional[np.ndarray] = None
    level_2_ids: Optional[np.ndarray] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "pair_id": self.unit_ids,
                "trajectory_type": self.cluster_labels,
            }
        )
        if self.level_1_ids is not None:
            df["level_1_id"] = self.level_1_ids
        if self.level_2_ids is not None:
            df["level_2_id"] = self.level_2_ids
        if self.representativeness is not None:
            rep = np.asarray(self.representativeness)
            if rep.ndim == 1:
                df["max_representativeness"] = rep
            else:
                for j in range(rep.shape[1]):
                    df[f"representativeness_{j + 1}"] = rep[:, j]
        return df

    def cluster_composition(self) -> pd.DataFrame:
        """Cross-tabulate trajectory types with level-1 and level-2 identifiers."""
        if self.level_1_ids is None or self.level_2_ids is None:
            raise ValueError(
                "level_1_ids and level_2_ids are required; pass them when building "
                "PairTypologyResult or use cluster_pair_trajectories()."
            )
        return pd.DataFrame(
            {
                "pair_id": self.unit_ids,
                "trajectory_type": self.cluster_labels,
                "level_1_id": self.level_1_ids,
                "level_2_id": self.level_2_ids,
            }
        )
