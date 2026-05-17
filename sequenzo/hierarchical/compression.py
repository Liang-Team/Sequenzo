"""
@Author  : 梁彧祺 Yuqi Liang
@File    : compression.py
@Time    : 13/05/2026 14:20
@Desc    :
    Aggregate identical pair-level relational trajectories with weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data import RelationalSequenceData, RelationalSequenceRecord


def _sequence_key(sequence: List[Any]) -> Tuple[Any, ...]:
    return tuple(sequence)


@dataclass
class CompressedRelationalSequences:
    """
    Unique pair trajectories with replication weights.

    ``compressed_data`` has one row per unique sequence pattern; ``weights`` align
    with those rows. ``pattern_to_pair_ids`` maps each pattern to all original pairs.
    """

    compressed_data: RelationalSequenceData
    weights: np.ndarray
    pattern_to_pair_ids: Dict[Tuple[Any, ...], List[str]]
    pattern_id_to_representative_pair: Dict[str, str]
    original_n_pairs: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.original_n_pairs == 0:
            return 1.0
        return float(self.compressed_data.n_pairs) / float(self.original_n_pairs)


def compress_identical_relational_sequences(
    sequence_data: RelationalSequenceData,
) -> CompressedRelationalSequences:
    """
    Collapse identical pair trajectories and attach occurrence weights.

    Each unique pattern receives a synthetic ``pattern_XXXXXX`` identifier.
    ``level_1_id`` / ``level_2_id`` on compressed rows come from the first
    observed pair and are illustrative only (patterns are sequence-identical,
    not level-identical). Use ``pattern_to_pair_ids`` for substantive pair ids.
    """
    pattern_map: Dict[Tuple[Any, ...], List[RelationalSequenceRecord]] = {}
    for rec in sequence_data.records:
        key = _sequence_key(rec.sequence)
        pattern_map.setdefault(key, []).append(rec)

    compressed_records: List[RelationalSequenceRecord] = []
    weights: List[float] = []
    pattern_to_pair_ids: Dict[Tuple[Any, ...], List[str]] = {}
    pattern_id_to_representative_pair: Dict[str, str] = {}

    for pattern_idx, (key, recs) in enumerate(pattern_map.items(), start=1):
        rep = recs[0]
        w = float(len(recs))
        pattern_id = f"pattern_{pattern_idx:06d}"
        pattern_to_pair_ids[key] = [r.pair_id for r in recs]
        pattern_id_to_representative_pair[pattern_id] = rep.pair_id
        compressed_records.append(
            RelationalSequenceRecord(
                pair_id=pattern_id,
                level_1_id=rep.level_1_id,
                level_2_id=rep.level_2_id,
                sequence=list(rep.sequence),
                time_points=list(rep.time_points),
                length=rep.length,
                n_missing=rep.n_missing,
            )
        )
        weights.append(w)

    compressed = RelationalSequenceData(
        records=compressed_records,
        level_1_col=sequence_data.level_1_col,
        level_2_col=sequence_data.level_2_col,
        time_col=sequence_data.time_col,
        state_col=sequence_data.state_col,
        pair_separator=sequence_data.pair_separator,
    )

    return CompressedRelationalSequences(
        compressed_data=compressed,
        weights=np.asarray(weights, dtype=float),
        pattern_to_pair_ids=pattern_to_pair_ids,
        pattern_id_to_representative_pair=pattern_id_to_representative_pair,
        original_n_pairs=sequence_data.n_pairs,
        details={
            "n_unique_patterns": len(compressed_records),
            "n_collapsed": sequence_data.n_pairs - len(compressed_records),
            "compressed_pattern_ids": True,
        },
    )


def expand_labels_to_original_pairs(
    compressed_labels: np.ndarray,
    compression: CompressedRelationalSequences,
    sequence_data: RelationalSequenceData,
) -> np.ndarray:
    """Map typology labels from compressed unique patterns to all original pairs."""
    pattern_to_label = {
        _sequence_key(rec.sequence): int(lab)
        for rec, lab in zip(compression.compressed_data.records, compressed_labels)
    }
    return np.asarray(
        [pattern_to_label[_sequence_key(rec.sequence)] for rec in sequence_data.records],
        dtype=int,
    )


def expand_rows_to_original_pairs(
    compressed_rows: np.ndarray,
    compression: CompressedRelationalSequences,
    sequence_data: RelationalSequenceData,
) -> np.ndarray:
    """Replicate per-pattern rows (e.g. distance-to-medoids) for all original pairs."""
    pattern_to_row = {
        _sequence_key(rec.sequence): compressed_rows[i]
        for i, rec in enumerate(compression.compressed_data.records)
    }
    return np.vstack(
        [pattern_to_row[_sequence_key(rec.sequence)] for rec in sequence_data.records]
    )
