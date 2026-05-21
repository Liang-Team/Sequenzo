"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqdist_mcse.py
@Time    : 19/05/2026 14:15
@Desc    : Pairwise MC standard errors of sequence dissimilarities.
"""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sequenzo.clustering.utils.aggregate_cases import aggregate_cases
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from ._params import parse_jprob, parse_kchanges, parse_model
from ._spell_convert import replicate_dss_dur_to_sequence_data
from ._utils import (
    dist_index,
    expand_unique_distances,
    single_spell_sequence,
    vector_to_dist,
)
from .mc_seq_replicate import combine_mc_replicate_list, mc_list_rep_set
from .mc_seqdist_se import DistMCResult, mc_ratios
from .timing_change import apply_timing_change


def _seq_mc_set(
    template: SequenceData,
    dss: np.ndarray,
    sdur: np.ndarray,
    row_indices: tuple[int, int],
    *,
    jprob: Union[int, float, np.ndarray],
    r: int,
    ch_meth: int,
    jfixed: bool,
    kchanges: Optional[int],
    rng: np.random.RandomState,
) -> SequenceData:
    """Build ``2*R`` sequences for one pair (R ``seqMCset``)."""
    i, j = row_indices
    frames = []
    for idx, label in ((i, "B1"), (j, "B2")):
        for rep in range(1, r + 1):
            new_dur = apply_timing_change(
                sdur[idx],
                ch_meth=ch_meth,
                jprob=jprob,
                jfixed=jfixed,
                kchanges=kchanges,
                rng=rng,
            )
            row_name = f"{rep}{label}"
            rep_one = replicate_dss_dur_to_sequence_data(
                template,
                dss[idx : idx + 1],
                new_dur[idx : idx + 1],
                [row_name],
            )
            frames.append(rep_one.data)
    combined = pd.concat(frames, ignore_index=True)
    return SequenceData(
        combined,
        time=template.time,
        states=template.states,
        labels=template.labels,
        id_col=template.id_col,
        start=template.start,
        void=template.void,
    )


def _pair_mc_mean_se(
    seq2sets: SequenceData,
    method: str,
    distance_kwargs: dict,
) -> tuple[float, float]:
    """Mean and SD of ``R`` cross-replicate distances (R ``seqdistsple``)."""
    n = len(seq2sets.seqdata)
    half = n // 2
    dists = get_distance_matrix(
        seq2sets,
        method=method,
        refseq=[list(range(half)), list(range(half, n))],
        full_matrix=False,
        **distance_kwargs,
    )
    dist_arr = np.asarray(dists, dtype=float).ravel()
    return float(np.mean(dist_arr)), float(np.std(dist_arr, ddof=1))


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero (-1 = all cores, 1 = sequential).")
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def _pair_seed(base_seed: int, position: int) -> int:
    return int((base_seed + position * 9973) % (2**31 - 1))


def _compute_one_pair(
    pos: int,
    i: int,
    j: int,
    *,
    work: SequenceData,
    dss: np.ndarray,
    sdur: np.ndarray,
    method: str,
    r: int,
    jprob: Union[int, float, np.ndarray],
    ch_meth: int,
    jfixed: bool,
    kchanges: Optional[int],
    distance_kwargs: dict,
    replset_combined: Optional[SequenceData],
    idbase: Optional[np.ndarray],
    pair_seed: int,
) -> Tuple[int, float, float]:
    """Worker: MC mean and SE for one sequence pair."""
    if single_spell_sequence(dss[i], sdur[i]) and single_spell_sequence(dss[j], sdur[j]):
        d_ij = get_distance_matrix(
            work,
            method=method,
            refseq=[i, j],
            full_matrix=False,
            **distance_kwargs,
        )
        return pos, float(np.asarray(d_ij).ravel()[0]), 0.0

    if replset_combined is not None and idbase is not None:
        pick = [int(idbase[rk] + i) for rk in range(r)] + [
            int(idbase[rk] + j) for rk in range(r)
        ]
        sub = replset_combined.data.iloc[pick].copy().reset_index(drop=True)
        seq2 = SequenceData(
            sub,
            time=work.time,
            states=work.states,
            labels=work.labels,
            id_col=work.id_col,
            start=work.start,
            void=work.void,
        )
    else:
        pair_rng = np.random.RandomState(pair_seed)
        seq2 = _seq_mc_set(
            work,
            dss,
            sdur,
            (i, j),
            jprob=jprob,
            r=r,
            ch_meth=ch_meth,
            jfixed=jfixed,
            kchanges=kchanges,
            rng=pair_rng,
        )
    m, s = _pair_mc_mean_se(seq2, method, distance_kwargs)
    return pos, m, s


def seqdist_mcse(
    seqdata: SequenceData,
    method: str = "LCS",
    J: Union[int, float, np.ndarray, list] = 1,
    R: int = 50,
    *,
    replic: str = "by.pair",
    unique: bool = True,
    model: str = "keep.dss",
    jfixed: bool = False,
    kchanges: Union[int, str, None] = None,
    ratios: bool = True,
    verbose: bool = True,
    n_jobs: int = 1,
    distance_kwargs: Optional[dict] = None,
    rng: Optional[Union[np.random.RandomState, int]] = None,
    random_engine: str = "numpy",
) -> DistMCResult:
    """
    MC mean and standard error of dissimilarities between sequence pairs (R ``seqdistMCSE``).

    For each pair ``(x, y)``, draws ``R`` timing-perturbed copies of each sequence and
    computes all ``R^2`` distances; returns the mean and sample SD across those draws.

    Parameters
    ----------
    replic
        ``"by.pair"`` (fresh replications per pair) or ``"once"`` (reuse one global set).
    unique
        If ``True``, compute on unique sequences and expand to all rows.
    n_jobs
        Parallel workers. ``1`` = sequential (default), ``-1`` = all CPU cores.
        Parallel mode requires ``random_engine="numpy"`` (R RNG is sequential only).
    random_engine
        ``"numpy"`` or ``"r"`` (R ``set.seed`` parity; forces ``n_jobs=1``).
    rng
        NumPy seed or ``RandomState`` for reproducibility when ``random_engine="numpy"``.
    """
    distance_kwargs = distance_kwargs or {}
    if replic not in ("by.pair", "once"):
        raise ValueError('replic must be "by.pair" or "once"')

    n_workers = _resolve_n_jobs(n_jobs)
    if random_engine == "r" and n_workers > 1:
        if verbose:
            print("[i] random_engine='r' requires sequential RNG; using n_jobs=1.")
        n_workers = 1

    ch_meth = parse_model(model)
    kchanges_i = parse_kchanges(kchanges)
    _, jprob = parse_jprob(J)

    if isinstance(rng, int):
        base_seed = rng
        rng = np.random.RandomState(rng)
    elif rng is None:
        base_seed = 1
        rng = np.random.RandomState(base_seed)
    else:
        base_seed = int(rng.randint(0, 2**31 - 1))

    if unique:
        agg = aggregate_cases(seqdata, weights=getattr(seqdata, "weights", None))
        u_idx = (agg.agg_index - 1).astype(int)
        u_frame = seqdata.data.iloc[u_idx].copy().reset_index(drop=True)
        u_seq = SequenceData(
            u_frame,
            time=seqdata.time,
            states=seqdata.states,
            labels=seqdata.labels,
            id_col=seqdata.id_col,
            weights=agg.agg_weights,
            start=seqdata.start,
            void=seqdata.void,
        )
        sdx = agg.disagg_index
        work = u_seq
    else:
        work = seqdata
        sdx = np.arange(1, len(seqdata.seqdata) + 1, dtype=int)

    dss = seqdss(work)
    sdur = seqdur(work)
    k = dss.shape[0]
    if k < 2:
        raise ValueError("Need at least two sequences.")

    labels = list(work.ids) if hasattr(work, "ids") else [str(i) for i in range(k)]
    kk = k * (k - 1) // 2
    mc_mean_vec = np.zeros(kk, dtype=float)
    mc_se_vec = np.zeros(kk, dtype=float)

    replset_combined = None
    idbase = None
    if replic == "once":
        rep_parts = mc_list_rep_set(
            work,
            jprob=jprob,
            r=R,
            ch_meth=ch_meth,
            jfixed=jfixed,
            kchanges=kchanges_i,
            rng=rng,
        )
        replset_combined = combine_mc_replicate_list(rep_parts)
        idbase = np.arange(0, R * k, k)

    pairs = [
        (dist_index(i + 1, j + 1, k) - 1, i, j)
        for i in range(k - 1)
        for j in range(i + 1, k)
    ]

    if verbose:
        print(
            f"[>] seqdist_mcse: {kk} pairs, R={R}, method={method!r}, "
            f"n_jobs={n_workers}, replic={replic!r}"
        )
        t0 = time.perf_counter()

    if n_workers == 1:
        for idx, (pos, i, j) in enumerate(pairs):
            _, m, s = _compute_one_pair(
                pos,
                i,
                j,
                work=work,
                dss=dss,
                sdur=sdur,
                method=method,
                r=R,
                jprob=jprob,
                ch_meth=ch_meth,
                jfixed=jfixed,
                kchanges=kchanges_i,
                distance_kwargs=distance_kwargs,
                replset_combined=replset_combined,
                idbase=idbase,
                pair_seed=_pair_seed(base_seed, pos),
            )
            mc_mean_vec[pos] = m
            mc_se_vec[pos] = s
            if verbose and (idx + 1) % max(1, kk // 10) == 0:
                print(f"    pairs done: {idx + 1}/{kk}", flush=True)
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, prefer="processes")(
            delayed(_compute_one_pair)(
                pos,
                i,
                j,
                work=work,
                dss=dss,
                sdur=sdur,
                method=method,
                r=R,
                jprob=jprob,
                ch_meth=ch_meth,
                jfixed=jfixed,
                kchanges=kchanges_i,
                distance_kwargs=distance_kwargs,
                replset_combined=replset_combined,
                idbase=idbase,
                pair_seed=_pair_seed(base_seed, pos),
            )
            for pos, i, j in pairs
        )
        for pos, m, s in results:
            mc_mean_vec[pos] = m
            mc_se_vec[pos] = s
        if verbose:
            print(f"    pairs done: {kk}/{kk}", flush=True)

    if verbose:
        print(f"[>] seqdist_mcse finished in {time.perf_counter() - t0:.1f}s")

    mc_mean_mat = vector_to_dist(mc_mean_vec, labels)
    mc_se_mat = vector_to_dist(mc_se_vec, labels)

    if unique and sdx is not None:
        orig_labels = list(seqdata.ids) if hasattr(seqdata, "ids") else [
            str(i) for i in range(len(seqdata.seqdata))
        ]
        mc_mean_mat, mc_se_mat = expand_unique_distances(
            mc_mean_mat, mc_se_mat, sdx, orig_labels
        )

    diss_o = get_distance_matrix(
        seqdata, method=method, full_matrix=True, **distance_kwargs
    )
    if isinstance(diss_o, pd.DataFrame):
        diss_o = diss_o.to_numpy(dtype=float)

    ret = DistMCResult(
        mc_mean=mc_mean_mat,
        mc_sd=mc_se_mat,
        mc_se=mc_se_mat,
        r=R,
        weights=getattr(seqdata, "weights", None),
        diss_o=diss_o,
        toref=False,
    )

    if ratios:
        ratio_res = mc_ratios(ret)
        ret.diss_z = ratio_res.diss_z
        ret.mc_mean_z = ratio_res.mc_mean_z
        ret.mean_se = ratio_res.mean_se
        ret.mc_se = ratio_res.mc_se

    return ret
