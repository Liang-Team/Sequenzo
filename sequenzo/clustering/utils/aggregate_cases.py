"""
@Author  : Yuqi Liang 梁彧祺
@File    : aggregate_cases.py
@Time    : 11/05/2025 18:22
@Desc    : 
Aggregate identical cases before weighted clustering.

Mirrors WeightedCluster ``wcAggregateCases``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

WC_SEPARATOR = "@@@WC_SEP@@"


@dataclass
class AggregateCasesResult:
    """Aggregated case table returned by :func:`aggregate_cases`."""

    agg_index: np.ndarray
    agg_weights: np.ndarray
    disagg_index: np.ndarray
    disagg_weights: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggIndex": self.agg_index,
            "aggWeights": self.agg_weights,
            "disaggIndex": self.disagg_index,
            "disaggWeights": self.disagg_weights,
        }


def _factorize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Recode each column independently, matching R ``factor`` level relabeling."""
    factored = frame.copy()
    for column in factored.columns:
        codes, _ = pd.factorize(factored[column], sort=False)
        factored[column] = (codes + 1).astype(str)
    return factored


def _aggregate_dataframe(frame: pd.DataFrame, weights: np.ndarray) -> AggregateCasesResult:
    factored = _factorize_columns(frame)
    row_keys = factored.astype(str).agg(WC_SEPARATOR.join, axis=1)

    mcorr = np.full(len(frame), np.nan, dtype=np.float64)
    agg_rows: list[tuple[int, float]] = []

    for _, group in pd.DataFrame({"key": row_keys, "index": np.arange(len(frame))}).groupby("key", sort=False):
        indices = group["index"].to_numpy(dtype=int)
        representative = int(indices[0])
        mcorr[indices] = representative
        agg_rows.append((representative, float(np.sum(weights[indices]))))

    agg_index = np.asarray([row[0] + 1 for row in agg_rows], dtype=int)
    agg_weights = np.asarray([row[1] for row in agg_rows], dtype=np.float64)
    disagg_index = np.array([np.where(agg_index == value)[0][0] + 1 for value in mcorr + 1], dtype=int)
    return AggregateCasesResult(
        agg_index=agg_index,
        agg_weights=agg_weights,
        disagg_index=disagg_index,
        disagg_weights=weights.copy(),
    )


def aggregate_cases(
    x: Union[pd.DataFrame, np.ndarray, Any],
    weights: Optional[np.ndarray] = None,
    *,
    weighted: bool = True,
) -> AggregateCasesResult:
    """
    Group identical rows and sum their weights.

    Returned indices follow WeightedCluster conventions: ``agg_index`` and
    ``disagg_index`` are 1-based.

    Parameters
    ----------
    x
        Input table. ``SequenceData`` objects use ``seqdata.seqdata``.
    weights
        Optional observation weights.
    weighted
        When ``x`` is sequence data and ``weights`` is omitted, read
        ``x.weights`` when available.
    """
    if hasattr(x, "seqdata"):
        frame = pd.DataFrame(x.seqdata)
        if weights is None and weighted:
            weights = getattr(x, "weights", None)
    else:
        frame = pd.DataFrame(x)

    if weights is None:
        weights = np.ones(len(frame), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != len(frame):
            raise ValueError("weights must have one value per row.")

    return _aggregate_dataframe(frame, weights)
