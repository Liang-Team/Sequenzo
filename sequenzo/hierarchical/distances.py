"""
@Author  : 梁彧祺 Yuqi Liang, Jan Meyerhoff-Liang
@File    : distances.py
@Time    : 05/04/2026 10:33
@Desc    :
    Pairwise sequence distances for relational (hierarchical) data.

    Distances are computed via existing Sequenzo :func:`~sequenzo.get_distance_matrix`
    machinery; this module attaches hierarchical metadata required for decomposition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .data import DEFAULT_MAX_FULL_MATRIX_PAIRS

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix

from .data import RelationalSequenceData


@dataclass
class RelationalDistanceMatrix:
    """
    Square distance matrix with hierarchical identifiers.

    Attributes
    ----------
    matrix : ndarray
        Symmetric pairwise distances, shape (n_pairs, n_pairs).
    pair_ids, level_1_ids, level_2_ids : ndarray
        Identifiers aligned with rows/columns of ``matrix``.
    method : str
        Distance method name passed to Sequenzo.
    representation : str
        ``"state"`` or ``"spell"`` (spell methods use Sequenzo spell distances).
    params : dict
        Extra arguments forwarded to :func:`get_distance_matrix`.
    seqdata : SequenceData, optional
        Underlying sequence object used for distance computation.
    """

    matrix: np.ndarray
    pair_ids: np.ndarray
    level_1_ids: np.ndarray
    level_2_ids: np.ndarray
    method: str
    representation: str = "state"
    params: Dict[str, Any] = field(default_factory=dict)
    seqdata: Optional[SequenceData] = None

    @property
    def n_pairs(self) -> int:
        return self.matrix.shape[0]

    def to_dataframe(self) -> pd.DataFrame:
        """Return distance matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.matrix,
            index=self.pair_ids,
            columns=self.pair_ids,
        )

    def as_numpy(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)


def relational_sequences_to_sequence_data(
    sequence_data: RelationalSequenceData,
    *,
    states: Optional[list] = None,
    labels: Optional[list] = None,
) -> SequenceData:
    """
    Build a :class:`~sequenzo.define_sequence_data.SequenceData` from pair sequences.

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    states : list, optional
        State alphabet. Inferred from data when omitted.
    labels : list, optional
        State labels for plotting; defaults to ``states``.
    """
    wide = sequence_data.to_wide_dataframe()
    time_cols = [str(t) for t in sequence_data.records[0].time_points]

    if states is None:
        states = sequence_data.states()
    if labels is None:
        labels = [str(s) for s in states]

    return SequenceData(
        data=wide,
        time=time_cols,
        states=states,
        labels=labels,
        id_col="pair_id",
    )


def compute_relational_distance_matrix(
    sequence_data: RelationalSequenceData,
    method: str = "HAM",
    representation: str = "state",
    *,
    states: Optional[list] = None,
    norm: str = "auto",
    full_matrix: bool = True,
    **distance_params: Any,
) -> RelationalDistanceMatrix:
    """
    Compute pairwise distances between all pair-level sequences.

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    method : str
        Sequenzo distance method (e.g. ``"HAM"``, ``"OM"``, ``"LCPspell"``).
    representation : str
        ``"state"`` for position-wise methods, or ``"spell"`` to prefer spell-based
        methods (``OMspell``, ``OMspellUnitFree``, ``LCPspell``, ``RLCPspell``).
        If ``representation="spell"`` and ``method`` is a generic name like ``"OM"``,
        ``method`` is mapped to ``"OMspell"``.
    states : list, optional
        Explicit state alphabet for :class:`SequenceData`.
    norm, full_matrix
        Forwarded to :func:`get_distance_matrix`.
    **distance_params
        Additional arguments (``sm``, ``indel``, ``expcost``, ``tpow``, etc.).

    Returns
    -------
    RelationalDistanceMatrix
    """
    representation = representation.lower().strip()
    if representation not in {"state", "spell"}:
        raise ValueError("representation must be 'state' or 'spell'.")

    effective_method = _resolve_method(method, representation)

    if sequence_data.n_pairs > DEFAULT_MAX_FULL_MATRIX_PAIRS:
        warnings.warn(
            f"Computing a full {sequence_data.n_pairs}×{sequence_data.n_pairs} distance "
            f"matrix may require ~{(sequence_data.n_pairs ** 2) * 8 / 1e9:.1f} GB in RAM. "
            "For large datasets, use sequenzo.hierarchical.sample_pairwise_distances() "
            "or sequence_discrepancy_by_level_sampled() instead.",
            UserWarning,
            stacklevel=2,
        )

    seqdata = relational_sequences_to_sequence_data(sequence_data, states=states)
    meta = sequence_data.to_dataframe().set_index("pair_id")

    params = dict(distance_params)
    om_needs_sm = {
        "OM",
        "OMspell",
        "OMspellUnitFree",
        "OMtspell",
        "OMstran",
        "OMloc",
        "OMslen",
    }
    if (
        representation == "spell"
        and effective_method in om_needs_sm
        and "sm" not in params
    ):
        from sequenzo.dissimilarity_measures import get_substitution_cost_matrix

        sm_result = get_substitution_cost_matrix(
            seqdata, method="CONSTANT", cval=1, miss_cost=1
        )
        sm = sm_result["sm"] if isinstance(sm_result, dict) else sm_result
        if hasattr(sm, "iloc"):
            sm = sm.iloc[1:, 1:].to_numpy(dtype=float)
        params["sm"] = np.asarray(sm, dtype=float)

    dist_result = get_distance_matrix(
        seqdata=seqdata,
        method=effective_method,
        norm=norm,
        full_matrix=full_matrix,
        **params,
    )

    if isinstance(dist_result, pd.DataFrame):
        matrix = dist_result.values.astype(float)
        pair_ids = dist_result.index.to_numpy(dtype=object)
        aligned_meta = meta.loc[pair_ids]
        level_1_ids = aligned_meta["level_1_id"].to_numpy(dtype=object)
        level_2_ids = aligned_meta["level_2_id"].to_numpy(dtype=object)
    else:
        matrix = np.asarray(dist_result, dtype=float)
        pair_ids = sequence_data.pair_ids
        level_1_ids = sequence_data.level_1_ids
        level_2_ids = sequence_data.level_2_ids

    if matrix.shape[0] != matrix.shape[1]:
        matrix = _ensure_square(matrix, len(pair_ids))

    return RelationalDistanceMatrix(
        matrix=matrix,
        pair_ids=pair_ids,
        level_1_ids=level_1_ids,
        level_2_ids=level_2_ids,
        method=effective_method,
        representation=representation,
        params={"norm": norm, **distance_params},
        seqdata=seqdata,
    )


def _resolve_method(method: str, representation: str) -> str:
    """Map generic method names to spell variants when requested."""
    if representation != "spell":
        return method

    spell_aliases = {
        "OM": "OMspell",
        "om": "OMspell",
        "LCP": "LCPspell",
        "lcp": "LCPspell",
        "RLCP": "RLCPspell",
        "rlcp": "RLCPspell",
    }
    return spell_aliases.get(method, method)


def _ensure_square(matrix: np.ndarray, n: int) -> np.ndarray:
    """Convert condensed distance vector to square matrix if needed."""
    if matrix.ndim == 1:
        from scipy.spatial.distance import squareform

        return squareform(matrix)
    if matrix.shape == (n, n):
        return matrix
    raise ValueError(
        f"Expected distance matrix of shape ({n}, {n}), got {matrix.shape}."
    )
