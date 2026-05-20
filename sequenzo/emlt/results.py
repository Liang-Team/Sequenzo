"""
Result container for EMLT (seqemlt) analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData


@dataclass
class EMLTResult:
    """
    Output of ``compute_emlt`` (TraMineRextras ``seqemlt``).

    Attributes mirror the R ``emlt`` list object.
    """

    states: np.ndarray
    period: int
    sit_time: np.ndarray
    situations: np.ndarray
    sit_states: np.ndarray
    sit_freq: pd.Series
    disjunctive: np.ndarray
    sit_transrate: pd.DataFrame
    sit_profil: pd.DataFrame
    distance_matrix: pd.DataFrame
    benz_covariance: np.ndarray
    pca: dict[str, Any]
    coord: np.ndarray
    sit_cor: pd.DataFrame
    seqdata: SequenceData
    a: float
    b: float
    weighted: bool

    @property
    def situation_labels_dot(self) -> list[str]:
        """Time-stamped situation labels with a dot separator (e.g. ``'1.5'``)."""
        return [f"{s}.{t}" for s, t in zip(self.sit_states, self.sit_time)]
