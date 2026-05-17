"""
@Author  : 梁彧祺 Yuqi Liang
@File    : typology.py
@Time    : 05/05/2026 19:47
@Desc    :
    Unified API for pair-level trajectory typology (PAM or CLARA backends).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..data import RelationalSequenceData
from ..distances import (
    RelationalDistanceMatrix,
    compute_relational_distance_matrix,
)
from .clara import cluster_pair_typology_clara
from .pam import cluster_pair_sequences
from .results import PairTypologyResult


def _pam_to_typology(
    pam_result,
    distance_matrix: RelationalDistanceMatrix,
) -> PairTypologyResult:
    matrix = distance_matrix.matrix
    medoids = np.asarray(pam_result.medoid_indices, dtype=int)
    distance_to_medoids = matrix[:, medoids]
    dmax = float(np.max(distance_to_medoids))
    representativeness = (
        1.0 - distance_to_medoids / dmax if dmax > 0 else np.ones_like(distance_to_medoids)
    )

    return PairTypologyResult(
        level="pair",
        k=pam_result.k,
        cluster_labels=np.asarray(pam_result.cluster_labels, dtype=int),
        unit_ids=np.asarray(pam_result.unit_ids, dtype=object),
        medoid_indices=medoids,
        medoid_ids=distance_matrix.pair_ids[medoids],
        distance_to_medoids=distance_to_medoids,
        method=pam_result.method,
        quality={"note": "full distance matrix PAM"},
        stability={},
        representativeness=representativeness,
        level_1_ids=distance_matrix.level_1_ids,
        level_2_ids=distance_matrix.level_2_ids,
        details={"pam_details": pam_result.details},
    )


def cluster_pair_trajectories(
    sequence_data: RelationalSequenceData,
    k: int,
    *,
    algorithm: str = "clara",
    distance_matrix: Optional[RelationalDistanceMatrix] = None,
    distance_method: str = "HAM",
    representation: str = "state",
    pam_method: str = "PAMonce",
    sample_size: Optional[int] = None,
    n_iterations: int = 100,
    clara_method: str = "crisp",
    criteria: Optional[List[str]] = None,
    stability: bool = True,
    max_dist: Optional[float] = None,
    distance_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
    aggregate_identical: bool = True,
) -> PairTypologyResult:
    """
    Identify scalable pair-level trajectory typology.

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    k : int
        Number of trajectory types (clusters).
    algorithm : str
        ``"clara"`` (default, scalable) or ``"pam"`` (requires full distance matrix).
    distance_matrix : RelationalDistanceMatrix, optional
        Precomputed distances; required for ``algorithm="pam"`` unless computed here.
    distance_method, representation
        Forwarded to distance computation when needed.
    sample_size, n_iterations, clara_method, criteria, stability, max_dist
        CLARA-specific options (see :func:`cluster_pair_typology_clara`).
    """
    algorithm = algorithm.lower().strip()
    if algorithm not in {"clara", "pam"}:
        raise ValueError("algorithm must be 'clara' or 'pam'.")

    if algorithm == "clara":
        return cluster_pair_typology_clara(
            sequence_data,
            k,
            distance_method=distance_method,
            representation=representation,
            sample_size=sample_size,
            n_iterations=n_iterations,
            clara_method=clara_method,
            criteria=criteria,
            stability=stability,
            max_dist=max_dist,
            distance_params=distance_params,
            random_state=random_state,
            verbose=verbose,
            aggregate_identical=aggregate_identical,
        )

    if distance_matrix is None:
        distance_matrix = compute_relational_distance_matrix(
            sequence_data,
            method=distance_method,
            representation=representation,
            **(distance_params or {}),
        )

    pam_result = cluster_pair_sequences(
        distance_matrix,
        k,
        method=pam_method,
        random_state=random_state,
        verbose=verbose,
    )
    return _pam_to_typology(pam_result, distance_matrix)
