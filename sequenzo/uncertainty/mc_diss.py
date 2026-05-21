"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_diss.py
@Time    : 11/05/2026 10:40
@Desc    : Replicate distance matrices and unique-distance extraction.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.clustering.utils.aggregate_cases import aggregate_cases
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
from .mc_seq_replicate import MCReplicateList


class MCDissList(list):
    """List of distance matrices with ``obs`` and ``toref`` metadata."""

    obs: bool
    toref: bool

    def __init__(
        self,
        items: List[Union[np.ndarray, pd.DataFrame]],
        *,
        obs: bool = False,
        toref: bool = False,
    ):
        super().__init__(items)
        self.obs = obs
        self.toref = toref


class UDistResult:
    """Dissimilarities between unique merged replicated sequences (R ``u.diss``)."""

    def __init__(
        self,
        matrix: Union[np.ndarray, pd.DataFrame],
        *,
        sdx: np.ndarray,
        n_sets: int,
        obs: bool,
        toref: bool,
    ):
        self.matrix = matrix
        self.sdx = sdx
        self.n_sets = n_sets
        self.obs = obs
        self.toref = toref


def _as_square_matrix(diss) -> np.ndarray:
    if isinstance(diss, pd.DataFrame):
        return diss.to_numpy(dtype=float)
    arr = np.asarray(diss, dtype=float)
    if arr.ndim == 1:
        from scipy.spatial.distance import squareform

        return squareform(arr)
    return arr


def _seqdist_to_ref(
    seqdata: SequenceData,
    ref: SequenceData,
    method: str,
    **kwargs: Any,
) -> np.ndarray:
    """Distances from ``seqdata`` rows to ``ref`` rows (R ``seqdistToRef``)."""
    nseq = len(seqdata.seqdata)
    nref = len(ref.seqdata)
    combined = pd.concat(
        [seqdata.data.reset_index(drop=True), ref.data.reset_index(drop=True)],
        ignore_index=True,
    )
    ref_ids = [f"ref.{rid}" for rid in ref.ids]
    if seqdata.id_col:
        combined.loc[nseq:, seqdata.id_col] = ref_ids
    combined_seq = SequenceData(
        combined,
        time=seqdata.time,
        states=seqdata.states,
        labels=seqdata.labels,
        id_col=seqdata.id_col,
        weights=getattr(seqdata, "weights", None),
        start=seqdata.start,
        void=seqdata.void,
    )
    return get_distance_matrix(
        combined_seq,
        method=method,
        refseq=[list(range(nseq)), list(range(nseq, nseq + nref))],
        full_matrix=True,
        **kwargs,
    )


def mc_udist(
    mc_r_seqdata: Sequence[SequenceData],
    method: str = "LCS",
    seqref: Optional[SequenceData] = None,
    **kwargs: Any,
) -> UDistResult:
    """
    Dissimilarities between unique merged replicated sequences (R ``MCudist``).
    """
    n_sets = len(mc_r_seqdata)
    frames = [s.data for s in mc_r_seqdata]
    combined = pd.concat(frames, ignore_index=True)
    weights = getattr(mc_r_seqdata[0], "weights", None)
    if weights is not None:
        weights = np.tile(np.asarray(weights).reshape(-1), n_sets)

    template = mc_r_seqdata[0]
    merged = SequenceData(
        combined,
        time=template.time,
        states=template.states,
        labels=template.labels,
        id_col=template.id_col,
        weights=weights,
        start=template.start,
        void=template.void,
    )
    agg = aggregate_cases(merged, weights=weights)
    u_idx = (agg.agg_index - 1).astype(int)
    u_frame = merged.data.iloc[u_idx].copy().reset_index(drop=True)
    u_seq = SequenceData(
        u_frame,
        time=template.time,
        states=template.states,
        labels=template.labels,
        id_col=template.id_col,
        weights=agg.agg_weights,
        start=template.start,
        void=template.void,
    )
    obs = False
    if isinstance(mc_r_seqdata, MCReplicateList):
        obs = mc_r_seqdata.obs

    if seqref is None:
        diss = get_distance_matrix(u_seq, method=method, full_matrix=True, **kwargs)
        toref = False
    else:
        diss = _seqdist_to_ref(u_seq, seqref, method=method, **kwargs)
        toref = True

    return UDistResult(
        _as_square_matrix(diss),
        sdx=agg.disagg_index,
        n_sets=n_sets,
        obs=obs,
        toref=toref,
    )


def mc_extract_dist(
    u_diss: UDistResult,
    k: int,
    *,
    full_matrix: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Extract the k-th replicated distance matrix (R ``MCExtractDist``)."""
    sdx = u_diss.sdx
    n_sets = u_diss.n_sets
    n = len(sdx) // n_sets
    if n * n_sets != len(sdx):
        raise ValueError("length(sdx) must be a multiple of N")
    idk = (np.arange(n) + (k - 1) * n).astype(int)
    sdx_k = (sdx[idk] - 1).astype(int)
    mat = u_diss.matrix
    if u_diss.toref:
        sub = mat[sdx_k, :]
        row_labels = [str(sdx[i]) for i in idk]
        return pd.DataFrame(sub, index=row_labels)
    sub = mat[np.ix_(sdx_k, sdx_k)]
    if full_matrix:
        labels = [str(sdx[i]) for i in idk]
        return pd.DataFrame(sub, index=labels, columns=labels)
    from scipy.spatial.distance import squareform

    return squareform(sub, checks=False)


def mc_nunique(
    mc_r_seqdata: Sequence[SequenceData],
    *,
    check: bool = False,
) -> Union[int, dict]:
    """Count unique replicated sequences (R ``MCnunique``)."""
    u = mc_udist(list(mc_r_seqdata), method="LCS")
    nu = u.matrix.shape[0]
    if not check:
        return nu
    n_sets = len(mc_r_seqdata)
    n = len(mc_r_seqdata[0].seqdata)
    u_ok = nu < n * np.sqrt(n_sets)
    return {"nu": nu, "u_ok": u_ok}


def mc_disslist(
    mc_r_seqdata: Sequence[SequenceData],
    method: str = "LCS",
    seqref: Optional[SequenceData] = None,
    *,
    full_matrix: bool = False,
    use_udiss: bool = False,
    **kwargs: Any,
) -> MCDissList:
    """
    Dissimilarity matrix for each MC-replicated dataset (R ``MCdisslist``).
    """
    obs = getattr(mc_r_seqdata, "obs", False) if hasattr(mc_r_seqdata, "obs") else False

    if seqref is None:
        if use_udiss:
            ud = mc_udist(mc_r_seqdata, method=method, **kwargs)
            disslist = [
                mc_extract_dist(ud, k, full_matrix=full_matrix)
                for k in range(1, ud.n_sets + 1)
            ]
            toref = False
        else:
            disslist = []
            for s in mc_r_seqdata:
                d = get_distance_matrix(s, method=method, full_matrix=full_matrix, **kwargs)
                disslist.append(_as_square_matrix(d) if full_matrix else d)
            toref = False
    else:
        if use_udiss:
            ud = mc_udist(mc_r_seqdata, method=method, seqref=seqref, **kwargs)
            disslist = [mc_extract_dist(ud, k) for k in range(1, ud.n_sets + 1)]
        else:
            disslist = [
                _seqdist_to_ref(s, seqref, method=method, **kwargs) for s in mc_r_seqdata
            ]
        toref = True

    return MCDissList(disslist, obs=obs, toref=toref)
