"""
@Author  : 梁彧祺 Yuqi Liang, 卫亚鹏 Yapeng Wei
@File    : clara.py
@Time    : 02/05/2026 09:12
@Desc    :
    CLARA-style scalable typology for pair-level relational trajectories.

    Adapts :func:`sequenzo.big_data.clara.clara` so clustered units are
    level-1 × level-2 pair sequences rather than individual sequences.
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sequenzo.big_data.clara.clara import clara
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix

from ..compression import CompressedRelationalSequences, compress_identical_relational_sequences
from ..data import RelationalSequenceData
from ..distances import (
    _resolve_method,
    relational_sequences_to_sequence_data,
)
from .results import PairTypologyResult


def _sequence_count(seqdata) -> int:
    if hasattr(seqdata, "n_sequences"):
        return int(seqdata.n_sequences)
    if hasattr(seqdata, "ids"):
        return len(seqdata.ids)
    return int(seqdata.values.shape[0])


def _default_max_dist(
    distance_method: str,
    dist_args: Dict[str, Any],
    *,
    sequence_length: Optional[int] = None,
) -> float:
    """Distance-scale maximum for representativeness (not state alphabet size)."""
    method = distance_method.upper()
    norm = str(dist_args.get("norm", "auto")).lower()
    if method in {"HAM", "DHD"} and norm in {"auto", "maxlength", "maxdist", "none"}:
        if norm == "none" and sequence_length:
            return float(sequence_length)
        return 1.0
    if sequence_length:
        return float(sequence_length)
    return 1.0


def _build_dist_args(
    sequence_data: RelationalSequenceData,
    *,
    distance_method: str,
    representation: str,
    distance_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    effective_method = _resolve_method(distance_method, representation.lower())
    params = dict(distance_params or {})
    dist_args: Dict[str, Any] = {
        "method": effective_method,
        "norm": params.pop("norm", "auto"),
        "full_matrix": True,
        **params,
    }

    om_needs_sm = {
        "OM",
        "OMspell",
        "OMspellRS",
        "OMtspell",
        "OMstran",
        "OMloc",
        "OMslen",
    }
    if (
        representation.lower() == "spell"
        and effective_method in om_needs_sm
        and "sm" not in dist_args
    ):
        from sequenzo.dissimilarity_measures import get_substitution_cost_matrix

        seqdata = relational_sequences_to_sequence_data(sequence_data)
        sm_result = get_substitution_cost_matrix(
            seqdata, method="CONSTANT", cval=1, miss_cost=1
        )
        sm = sm_result["sm"] if isinstance(sm_result, dict) else sm_result
        if hasattr(sm, "iloc"):
            sm = sm.iloc[1:, 1:].to_numpy(dtype=float)
        dist_args["sm"] = np.asarray(sm, dtype=float)

    return dist_args


def _distances_to_medoids(
    seqdata,
    medoid_indices: np.ndarray,
    dist_args: Dict[str, Any],
) -> np.ndarray:
    """Return n_sequences × k distances to medoid rows (0-based indices)."""
    medoid_indices = np.asarray(medoid_indices, dtype=int).reshape(-1)
    n = _sequence_count(seqdata)
    refseq = [list(range(n)), medoid_indices.tolist()]
    matrix_args = dict(dist_args)
    matrix_args.pop("full_matrix", None)
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            diss = get_distance_matrix(
                seqdata=seqdata,
                refseq=refseq,
                full_matrix=False,
                **matrix_args,
            )
    return diss.to_numpy(dtype=float)


def cluster_pair_typology_clara(
    sequence_data: RelationalSequenceData,
    k: int,
    *,
    distance_method: str = "HAM",
    representation: str = "state",
    sample_size: Optional[int] = None,
    n_iterations: int = 100,
    clara_method: str = "crisp",
    criteria: Optional[List[str]] = None,
    stability: bool = True,
    max_dist: Optional[Union[float, str]] = None,
    distance_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    verbose: bool = True,
    aggregate_identical: bool = True,
) -> PairTypologyResult:
    """
    Scalable pair-level trajectory typology via CLARA-style clustering.

    Repeatedly subsamples pair trajectories, runs PAM (or fuzzy / representativeness
    variants) on the subsample, and assigns all pairs to medoids using only
    n × K distances to representative trajectories.
    """
    if k < 2:
        raise ValueError("k must be at least 2 for CLARA typology.")

    if random_state is not None:
        np.random.seed(random_state)

    compression: Optional[CompressedRelationalSequences] = None
    work_data = sequence_data
    if aggregate_identical:
        candidate = compress_identical_relational_sequences(sequence_data)
        if candidate.compressed_data.n_pairs > k:
            compression = candidate
            work_data = compression.compressed_data

    n_pairs = work_data.n_pairs
    if sample_size is None:
        sample_size = min(5000, max(40 + 2 * k, int(np.ceil(0.1 * n_pairs))))
    sample_size = int(sample_size)
    if sample_size < k:
        raise ValueError("sample_size must be at least k.")

    seqdata = relational_sequences_to_sequence_data(work_data)
    dist_args = _build_dist_args(
        work_data,
        distance_method=distance_method,
        representation=representation,
        distance_params=distance_params,
    )

    seq_len = work_data.records[0].length if work_data.records else None
    if clara_method.lower() == "representativeness":
        if max_dist is None or max_dist == "auto":
            max_dist = _default_max_dist(distance_method, dist_args, sequence_length=seq_len)
        max_dist = float(max_dist)

    if not verbose:
        import io
        import sys

        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            clara_result = clara(
                seqdata,
                R=n_iterations,
                kvals=[k],
                sample_size=sample_size,
                method=clara_method.lower(),
                dist_args=dist_args,
                criteria=criteria or ["distance"],
                stability=stability,
                max_dist=max_dist,
            )
        finally:
            sys.stdout = _stdout
    else:
        clara_result = clara(
            seqdata,
            R=n_iterations,
            kvals=[k],
            sample_size=sample_size,
            method=clara_method.lower(),
            dist_args=dist_args,
            criteria=criteria or ["distance"],
            stability=stability,
            max_dist=max_dist,
        )

    typology = _clara_output_to_typology(
        clara_result,
        k=k,
        seqdata=seqdata,
        dist_args=dist_args,
        clara_method=clara_method.lower(),
        sample_size=sample_size,
        n_iterations=n_iterations,
        level_1_ids=work_data.level_1_ids,
        level_2_ids=work_data.level_2_ids,
        weights=compression.weights if compression is not None else None,
        compression=compression,
        original_sequence_data=sequence_data,
        representativeness_max_dist=(
            float(max_dist) if clara_method.lower() == "representativeness" else None
        ),
    )
    return typology


def _clara_output_to_typology(
    clara_result: Dict[str, Any],
    *,
    k: int,
    seqdata,
    dist_args: Dict[str, Any],
    clara_method: str,
    sample_size: int,
    n_iterations: int,
    level_1_ids: np.ndarray,
    level_2_ids: np.ndarray,
    weights: Optional[np.ndarray],
    compression: Optional[CompressedRelationalSequences],
    original_sequence_data: RelationalSequenceData,
    representativeness_max_dist: Optional[float] = None,
) -> PairTypologyResult:
    def _matrix_from_public_cells(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        if values.ndim == 1:
            rows = [np.asarray(value, dtype=float).reshape(-1) for value in values]
            if not rows:
                return np.empty((0, 0), dtype=float)
            width = rows[0].size
            if any(row.size != width for row in rows):
                raise ValueError("CLARA matrix-valued output has inconsistent row widths")
            return np.vstack(rows)
        return np.asarray(values, dtype=float)

    k_idx = 0
    best = clara_result["clara"][k_idx]
    stats_row = clara_result["stats"].iloc[k_idx]

    raw_labels = clara_result["clustering"].iloc[:, k_idx].to_numpy()
    pair_ids = np.asarray(seqdata.ids, dtype=object)

    if clara_method in ("fuzzy", "noise"):
        membership = _matrix_from_public_cells(raw_labels)
        cluster_labels = np.argmax(membership, axis=1)
        representativeness = None
    elif clara_method == "representativeness":
        representativeness = _matrix_from_public_cells(raw_labels)
        membership = None
        cluster_labels = np.argmax(representativeness, axis=1)
    else:
        cluster_labels = np.asarray(raw_labels, dtype=int) - 1
        membership = None
        representativeness = best.get("representativeness")

    # CLARA ``best["medoids"]`` are 0-based row indices (see big_data/clara/clara.py).
    medoid_rows = np.asarray(best["medoids"], dtype=int).reshape(-1)
    medoid_rows = np.clip(medoid_rows, 0, max(len(pair_ids) - 1, 0))
    medoid_ids = pair_ids[medoid_rows]
    compressed_medoid_rows = medoid_rows.copy() if compression is not None else None
    compressed_medoid_ids = medoid_ids.copy() if compression is not None else None

    if representativeness is None or not hasattr(representativeness, "ndim") or representativeness.ndim != 2:
        distance_to_medoids = _distances_to_medoids(seqdata, medoid_rows, dist_args)
    else:
        rep_scale = 1.0 if representativeness_max_dist is None else float(representativeness_max_dist)
        distance_to_medoids = (1.0 - representativeness) * rep_scale

    quality = {
        "avg_dist": float(stats_row["Avg dist"]),
        "pbm": float(stats_row["PBM"]),
        "db": float(stats_row["DB"]),
        "xb": float(stats_row["XB"]),
        "ams": float(stats_row["AMS"]),
        "best_iteration": int(stats_row["Best iter"]),
        "objective": float(best.get("objective", np.nan)),
    }
    stability = {
        "ari_above_0.8": float(stats_row["ARI>0.8"]),
        "jc_above_0.8": float(stats_row["JC>0.8"]),
        "n_iterations": n_iterations,
    }
    if best.get("arimatrix") is not None and not isinstance(best["arimatrix"], float):
        stability["arimatrix"] = best["arimatrix"]

    if representativeness is None and distance_to_medoids is not None:
        dmax = float(np.max(distance_to_medoids))
        if dmax > 0:
            representativeness = 1.0 - distance_to_medoids / dmax

    full_l1 = original_sequence_data.level_1_ids
    full_l2 = original_sequence_data.level_2_ids
    full_unit_ids = original_sequence_data.pair_ids

    if compression is not None:
        from ..compression import (
            expand_labels_to_original_pairs,
            expand_rows_to_original_pairs,
        )

        cluster_labels = expand_labels_to_original_pairs(
            cluster_labels, compression, original_sequence_data
        )
        if distance_to_medoids is not None:
            distance_to_medoids = expand_rows_to_original_pairs(
                distance_to_medoids, compression, original_sequence_data
            )
        if representativeness is not None and representativeness.ndim == 2:
            representativeness = expand_rows_to_original_pairs(
                representativeness, compression, original_sequence_data
            )
        if membership is not None and membership.ndim == 2:
            membership = expand_rows_to_original_pairs(
                membership, compression, original_sequence_data
            )
        representative_lookup = compression.pattern_id_to_representative_pair
        medoid_ids = np.asarray(
            [representative_lookup.get(str(pattern_id), pattern_id) for pattern_id in medoid_ids],
            dtype=object,
        )
        original_index = {pair_id: idx for idx, pair_id in enumerate(full_unit_ids)}
        try:
            medoid_rows = np.asarray([original_index[pair_id] for pair_id in medoid_ids], dtype=int)
        except KeyError as exc:
            raise ValueError("Compressed CLARA medoid IDs must map to original pair IDs") from exc

    return PairTypologyResult(
        level="pair",
        k=k,
        cluster_labels=cluster_labels,
        unit_ids=full_unit_ids,
        medoid_indices=medoid_rows,
        medoid_ids=medoid_ids,
        distance_to_medoids=distance_to_medoids,
        method="CLARA",
        quality=quality,
        stability=stability,
        membership=membership if clara_method in ("fuzzy", "noise") else None,
        representativeness=representativeness,
        weights=weights,
        level_1_ids=full_l1,
        level_2_ids=full_l2,
        details={
            "clara_method": clara_method,
            "sample_size": sample_size,
            "compression": compression.details if compression is not None else None,
            "pattern_id_to_representative_pair": (
                compression.pattern_id_to_representative_pair
                if compression is not None
                else None
            ),
            "compressed_medoid_indices": compressed_medoid_rows,
            "compressed_medoid_ids": compressed_medoid_ids,
            "representativeness_max_dist": representativeness_max_dist,
            "evol_diss": best.get("evol_diss"),
        },
    )
