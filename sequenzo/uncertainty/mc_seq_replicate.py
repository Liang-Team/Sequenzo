"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_seq_replicate.py
@Time    : 09/05/2026 21:50
@Desc    : Monte Carlo timing-perturbed sequence replication.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from sequenzo.clustering.utils.aggregate_cases import aggregate_cases
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from ._params import parse_jprob, parse_kchanges, parse_model
from ._spell_convert import replicate_dss_dur_to_sequence_data
from .timing_change import apply_timing_change


class MCReplicateList(list):
    """List of replicated ``SequenceData`` with R-style attributes."""

    unique: bool
    obs: bool
    sdx: Optional[np.ndarray]

    def __init__(
        self,
        items: List[SequenceData],
        *,
        unique: bool,
        obs: bool,
        sdx: Optional[np.ndarray] = None,
    ):
        super().__init__(items)
        self.unique = unique
        self.obs = obs
        self.sdx = sdx


def _alter_all_durations(
    sdur: np.ndarray,
    *,
    ch_meth: int,
    jprob: Union[int, float, np.ndarray],
    jfixed: bool,
    kchanges: Optional[int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply timing changes row-wise; return new duration matrix."""
    n_rows = sdur.shape[0]
    out = np.zeros_like(sdur)
    for r in range(n_rows):
        out[r] = apply_timing_change(
            sdur[r],
            ch_meth=ch_meth,
            jprob=jprob,
            jfixed=jfixed,
            kchanges=kchanges,
            rng=rng,
        )
    return out


def mc_list_rep_set(
    seqdata: SequenceData,
    *,
    jprob: Union[int, float, np.ndarray],
    r: int = 10,
    ch_meth: int = 1,
    jfixed: bool = False,
    kchanges: Optional[int] = None,
    rng=None,
) -> List[SequenceData]:
    """
    Build ``R`` replicated ``SequenceData`` objects from unique sequences in ``seqdata``.

    Internal counterpart of R ``MClistrepset``.
    """
    rng = rng or np.random.RandomState()
    dss = seqdss(seqdata)
    sdur = seqdur(seqdata)
    n_rows = dss.shape[0]
    base_ids = list(seqdata.ids) if hasattr(seqdata, "ids") else [str(i) for i in range(n_rows)]
    weights = getattr(seqdata, "weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)

    result: List[SequenceData] = []
    for rep in range(1, r + 1):
        new_dur = _alter_all_durations(
            sdur,
            ch_meth=ch_meth,
            jprob=jprob,
            jfixed=jfixed,
            kchanges=kchanges,
            rng=rng,
        )
        row_names = [f"R{rep}-{bid}" for bid in base_ids]
        rep_seq = replicate_dss_dur_to_sequence_data(
            seqdata, dss, new_dur, row_names, weights=weights
        )
        result.append(rep_seq)
    return result


def mc_seq_replicate(
    seqdata: SequenceData,
    J: Union[int, float, np.ndarray, list] = 1,
    R: int = 20,
    *,
    unique: bool = False,
    model: str = "keep.dss",
    jfixed: bool = False,
    kchanges: Union[int, str, None] = None,
    include_obs: bool = False,
    rng: Optional[np.random.RandomState] = None,
    random_engine: str = "numpy",
) -> MCReplicateList:
    """
    Generate ``R`` Monte Carlo altered sequence datasets (R ``MCseqReplicate``).

    Parameters
    ----------
    seqdata
        Input state sequences.
    J
        Max timing error (integer) or odd-length probability vector over
        ``{-K, ..., 0, ..., K}``.
    R
        Number of replicated datasets.
    unique
        If ``True``, replicate only unique sequences and propagate weights.
    model
        ``"keep.dss"``, ``"indep"``, or ``"relative"``.
    jfixed
        Use the same random shift for all selected transitions in a sequence.
    kchanges
        Number of transitions to perturb; ``None`` = all; ``"rand"`` = random count.
    include_obs
        Append observed (unique) data as last list element.
    rng
        NumPy ``RandomState`` when ``random_engine='numpy'``.
    random_engine
        ``'numpy'`` (default) or ``'r'`` for R ``set.seed``-compatible draws (requires R).

    Returns
    -------
    list
        ``R`` (or ``R+1``) ``SequenceData`` objects. Attributes ``unique`` and ``obs``
        are stored on the list object as dict entries ``_mc_meta``.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("seqdata must be a SequenceData object.")

    ch_meth = parse_model(model)
    kchanges_i = parse_kchanges(kchanges)
    _, jprob = parse_jprob(J)
    if isinstance(rng, int):
        seed = int(rng)
        if random_engine == "r":
            from .r_random import r_random_state

            rng = r_random_state(seed)
        else:
            rng = np.random.RandomState(seed)
    elif rng is None:
        if random_engine == "r":
            from .r_random import r_random_state

            rng = r_random_state(1)
        else:
            rng = np.random.RandomState()

    if unique:
        agg = aggregate_cases(seqdata, weights=getattr(seqdata, "weights", None))
        agg_idx = (agg.agg_index - 1).astype(int)
        u_frame = seqdata.data.iloc[agg_idx].copy().reset_index(drop=True)
        u_seq = SequenceData(
            u_frame,
            time=seqdata.time,
            states=seqdata.states,
            labels=seqdata.labels,
            id_col=seqdata.id_col,
            weights=agg.agg_weights,
            start=seqdata.start,
            missing_values=getattr(seqdata, "missing_values", None),
            void=seqdata.void,
        )
        work = u_seq
        sdx = agg.disagg_index
    else:
        work = seqdata
        sdx = None

    rep_items = mc_list_rep_set(
        work,
        jprob=jprob,
        r=R,
        ch_meth=ch_meth,
        jfixed=jfixed,
        kchanges=kchanges_i,
        rng=rng,
    )

    if include_obs:
        rep_items.append(work)

    return MCReplicateList(
        rep_items,
        unique=unique,
        obs=include_obs,
        sdx=sdx,
    )


def combine_mc_replicate_list(rep_list: List[SequenceData]) -> SequenceData:
    """Stack replicated datasets into one object (R ``rbind`` of MC sets)."""
    template = rep_list[0]
    combined = pd.concat([s.data for s in rep_list], ignore_index=True)
    weights = getattr(template, "weights", None)
    if weights is not None:
        weights = np.tile(np.asarray(weights).reshape(-1), len(rep_list))
    return SequenceData(
        combined,
        time=template.time,
        states=template.states,
        labels=template.labels,
        id_col=template.id_col,
        weights=weights,
        start=template.start,
        void=template.void,
    )
