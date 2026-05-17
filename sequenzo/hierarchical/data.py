"""
@Author  : 梁彧祺 Yuqi Liang, Jan Meyerhoff-Liang
@File    : data.py
@Time    : 02/04/2026 08:42
@Desc    :
    Data preparation for hierarchical (relational) sequence analysis.

    Each row in the input is one observation of one relational unit (pair of
    level-1 and level-2 entities) at one time point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DEFAULT_PAIR_SEPARATOR = "__"
# Full n×n distance matrices above this size are discouraged (memory ~ n² × 8 bytes).
DEFAULT_MAX_FULL_MATRIX_PAIRS = 8_000


@dataclass
class RelationalSequenceRecord:
    """One pair-level trajectory with metadata."""

    pair_id: str
    level_1_id: Any
    level_2_id: Any
    sequence: List[Any]
    time_points: List[Any]
    length: int
    n_missing: int = 0


@dataclass
class RelationalSequenceData:
    """
    Container for pair-level sequences and hierarchical identifiers.

    Attributes
    ----------
    records : list of RelationalSequenceRecord
    level_1_col, level_2_col, time_col, state_col : str
        Original column names from the long-format input.
    """

    records: List[RelationalSequenceRecord]
    level_1_col: str
    level_2_col: str
    time_col: str
    state_col: str
    pair_separator: str = DEFAULT_PAIR_SEPARATOR

    @property
    def n_pairs(self) -> int:
        return len(self.records)

    @property
    def pair_ids(self) -> np.ndarray:
        return np.array([r.pair_id for r in self.records], dtype=object)

    @property
    def level_1_ids(self) -> np.ndarray:
        return np.array([r.level_1_id for r in self.records], dtype=object)

    @property
    def level_2_ids(self) -> np.ndarray:
        return np.array([r.level_2_id for r in self.records], dtype=object)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tabular view of pair-level sequences."""
        return pd.DataFrame(
            {
                "pair_id": self.pair_ids,
                "level_1_id": self.level_1_ids,
                "level_2_id": self.level_2_ids,
                "sequence": [r.sequence for r in self.records],
                "time_points": [r.time_points for r in self.records],
                "length": [r.length for r in self.records],
                "n_missing": [r.n_missing for r in self.records],
            }
        )

    def to_wide_dataframe(self) -> pd.DataFrame:
        """
        Wide format suitable for :class:`~sequenzo.define_sequence_data.SequenceData`.

        Rows are pairs; columns are ``pair_id``, ``level_1_id``, ``level_2_id``,
        and one column per time point (string column names).
        """
        if not self.records:
            raise ValueError(
                "Cannot build wide dataframe: RelationalSequenceData has no pair "
                "records. Use make_relational_sequences() on non-empty long-format data."
            )
        time_cols = [str(t) for t in self.records[0].time_points]
        rows = []
        for rec in self.records:
            row = {
                "pair_id": rec.pair_id,
                "level_1_id": rec.level_1_id,
                "level_2_id": rec.level_2_id,
            }
            for t_col, state in zip(time_cols, rec.sequence):
                row[t_col] = state
            rows.append(row)
        return pd.DataFrame(rows)

    def states(self) -> List[Any]:
        """Sorted unique states across all pair sequences."""
        seen = set()
        for rec in self.records:
            for s in rec.sequence:
                if pd.isna(s):
                    continue
                seen.add(s)
        return sorted(seen, key=lambda x: str(x))


def make_pair_id(
    level_1_id: Any,
    level_2_id: Any,
    separator: str = DEFAULT_PAIR_SEPARATOR,
) -> str:
    """Build a stable pair identifier from level-1 and level-2 IDs."""
    return f"{level_1_id}{separator}{level_2_id}"


def validate_relational_sequence_data(
    data: pd.DataFrame,
    level_1_col: str,
    level_2_col: str,
    time_col: str,
    state_col: str,
    *,
    require_balanced: bool = True,
    pair_separator: str = DEFAULT_PAIR_SEPARATOR,
) -> Dict[str, Any]:
    """
    Validate long-format relational sequence data.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format data with one row per pair-time observation.
    level_1_col, level_2_col, time_col, state_col : str
        Column names for the two hierarchical levels, time, and state.
    require_balanced : bool
        If True, raise when pairs have unequal numbers of time points.
    pair_separator : str
        Separator used when constructing pair IDs.

    Returns
    -------
    dict
        Summary statistics (counts, balance, missing rate).

    Raises
    ------
    ValueError
        When required columns are missing or data fail structural checks.
    """
    required = [level_1_col, level_2_col, time_col, state_col]
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {required}."
        )

    df = data[required].copy()
    n_rows = len(df)

    if df[level_1_col].isna().any() or df[level_2_col].isna().any():
        raise ValueError("Missing values in level-1 or level-2 ID columns.")

    if df[time_col].isna().any():
        raise ValueError(f"Missing values in time column '{time_col}'.")

    dup_mask = df.duplicated(subset=[level_1_col, level_2_col, time_col], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(
            f"Duplicate pair-time observations found ({n_dup} rows). "
            "Each (level_1, level_2, time) combination must appear at most once."
        )

    df["_pair_id"] = df.apply(
        lambda r: make_pair_id(r[level_1_col], r[level_2_col], pair_separator),
        axis=1,
    )

    lengths = df.groupby("_pair_id", sort=False)[time_col].nunique()
    length_dist = lengths.value_counts().sort_index()
    balanced = len(length_dist) == 1

    if require_balanced and not balanced:
        raise ValueError(
            "Sequences are not balanced across pairs. "
            f"Time-point counts per pair: {length_dist.to_dict()}. "
            "Align, pad, or trim data before analysis, or set require_balanced=False."
        )

    same_time_grid = True
    if require_balanced and balanced and n_rows > 0:
        time_sets = df.groupby("_pair_id", sort=False)[time_col].apply(
            lambda x: tuple(x.sort_values().tolist())
        )
        common_grid = time_sets.iloc[0]
        same_time_grid = bool(time_sets.apply(lambda x: x == common_grid).all())
        if not same_time_grid:
            raise ValueError(
                "Pairs have the same number of time points but not the same time grid. "
                "Each pair must observe the exact same time points so that sequence "
                "positions are comparable across pairs."
            )

    n_level_1 = df[level_1_col].nunique()
    n_level_2 = df[level_2_col].nunique()
    n_pairs = df["_pair_id"].nunique()
    n_time_points = int(lengths.iloc[0]) if balanced and len(lengths) else int(lengths.max())

    state_missing = df[state_col].isna().sum()
    missing_rate = float(state_missing / n_rows) if n_rows else 0.0

    return {
        "n_level_1": int(n_level_1),
        "n_level_2": int(n_level_2),
        "n_pairs": int(n_pairs),
        "n_time_points": n_time_points,
        "balanced": bool(balanced),
        "same_time_grid": bool(same_time_grid),
        "missing_rate": missing_rate,
        "sequence_length_distribution": length_dist.to_dict(),
        "n_rows": n_rows,
    }


def check_balanced_panel(
    data: pd.DataFrame,
    level_1_col: str,
    level_2_col: str,
    time_col: str,
    *,
    require_same_time_grid: bool = True,
) -> bool:
    """Return True if every pair shares the same time-point count and time grid."""
    pair_key = (
        data[level_1_col].astype(str)
        + DEFAULT_PAIR_SEPARATOR
        + data[level_2_col].astype(str)
    )
    lengths = data.groupby(pair_key)[time_col].nunique()
    if len(lengths.value_counts()) != 1:
        return False
    if not require_same_time_grid:
        return True
    time_sets = data.groupby(pair_key)[time_col].apply(
        lambda x: tuple(x.sort_values().tolist())
    )
    return bool(time_sets.apply(lambda x: x == time_sets.iloc[0]).all())


def make_relational_sequences(
    data: pd.DataFrame,
    level_1_col: str = "level_1_id",
    level_2_col: str = "level_2_id",
    time_col: str = "time",
    state_col: str = "state",
    *,
    validate: bool = True,
    require_balanced: bool = True,
    pair_separator: str = DEFAULT_PAIR_SEPARATOR,
    sort_time: bool = True,
) -> RelationalSequenceData:
    """
    Convert long-format relational data to one sequence per pair.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format input.
    level_1_col, level_2_col, time_col, state_col : str
        Column names.
    validate : bool
        Run :func:`validate_relational_sequence_data` before conversion.
    require_balanced : bool
        Passed to validation when ``validate=True``.
    pair_separator : str
        Separator between level-1 and level-2 IDs in ``pair_id``.
    sort_time : bool
        Sort time points within each pair before building the sequence.

    Returns
    -------
    RelationalSequenceData
    """
    if validate:
        validate_relational_sequence_data(
            data,
            level_1_col,
            level_2_col,
            time_col,
            state_col,
            require_balanced=require_balanced,
            pair_separator=pair_separator,
        )

    df = data[[level_1_col, level_2_col, time_col, state_col]].copy()
    records: List[RelationalSequenceRecord] = []

    grouped = df.groupby([level_1_col, level_2_col], sort=False)
    for (l1, l2), grp in grouped:
        grp = grp.sort_values(time_col) if sort_time else grp
        times = grp[time_col].tolist()
        states = grp[state_col].tolist()
        n_missing = int(grp[state_col].isna().sum())
        pair_id = make_pair_id(l1, l2, pair_separator)
        records.append(
            RelationalSequenceRecord(
                pair_id=pair_id,
                level_1_id=l1,
                level_2_id=l2,
                sequence=states,
                time_points=times,
                length=len(states),
                n_missing=n_missing,
            )
        )

    return RelationalSequenceData(
        records=records,
        level_1_col=level_1_col,
        level_2_col=level_2_col,
        time_col=time_col,
        state_col=state_col,
        pair_separator=pair_separator,
    )
