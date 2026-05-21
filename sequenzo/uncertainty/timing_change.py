"""
@Author  : Yuqi Liang 梁彧祺
@File    : timing_change.py
@Time    : 06/05/2026 15:45
@Desc    : Spell duration perturbation models (keep.dss, indep, relative).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ._params import parse_jprob


def _move(n: int, jprob: Union[int, float, np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    """
    Draw ``n`` timing shifts from the discrete distribution defined by ``jprob``.

    Mirrors R ``move()``: scalar ``J`` uses uniform shifts in ``[-J, J]``;
    a probability vector uses inverse-CDF sampling with zero-probability bins removed.
    """
    if isinstance(jprob, np.ndarray) and jprob.ndim == 0:
        jprob = int(jprob)
    if np.isscalar(jprob) or (isinstance(jprob, (int, float)) and not isinstance(jprob, bool)):
        j_int = int(jprob)
        # floor(runif(n) * (2*J + 1) - J)
        u = rng.random(n)
        return np.floor(u * (2 * j_int + 1) - j_int).astype(np.int64)
    probs = np.asarray(jprob, dtype=float)
    probs = probs / probs.sum()
    jj = probs.size
    nonzero = np.flatnonzero(probs != 0)
    probs_nz = probs[nonzero]
    cdf = np.cumsum(probs_nz)
    jl = cdf.size
    p = rng.random(n)
    res = np.empty(n, dtype=np.int64)
    for idx, u in enumerate(p):
        hit = np.flatnonzero(cdf >= u)
        r = hit[0] if hit.size else jl - 1
        if r <= 0:
            r = 0
        if r > jl - 1:
            r = jl - 1
        res[idx] = nonzero[r] - (jj + 1) // 2
    return res


def _set_klist(
    k: Optional[int],
    nt: int,
    rng,
) -> np.ndarray:
    """Indices of transitions that may receive a timing error (1-based spell indices)."""
    if k is None:
        k_eff = nt
    elif k < 0:
        # floor(runif(1, min=0, max=nt) + 0.5) — R max=nt is exclusive
        u = float(rng.random(1)[0])
        k_eff = int(np.floor(u * nt + 0.5))
    else:
        k_eff = k
    if k_eff < nt:
        if hasattr(rng, "sample_int"):
            return rng.sample_int(nt, k_eff).astype(np.int64)
        chosen = rng.choice(nt, size=k_eff, replace=False)
        return (chosen + 1).astype(np.int64)
    return np.arange(1, nt + 1, dtype=np.int64)


def _count_transitions(sduri: np.ndarray) -> int:
    """Number of transitions = number of positive-duration spells minus one."""
    return int(np.sum(sduri > 0)) - 1


def ch_dur(
    sduri: np.ndarray,
    jprob: Union[int, float, np.ndarray],
    *,
    jfixed: bool = False,
    k: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Model ``keep.dss``: shift transition times between adjacent spells only.

    Preserves the distinct successive state sequence (DSS); total sequence length unchanged.
    """
    rng = rng or np.random.RandomState()
    out = np.array(sduri, dtype=np.int64, copy=True)
    nt = _count_transitions(out)
    if nt <= 0:
        return out
    klist = _set_klist(k, nt, rng)
    change = np.zeros(nt, dtype=np.int64)
    if jfixed:
        change[klist - 1] = _move(1, jprob, rng)
    else:
        change[klist - 1] = _move(len(klist), jprob, rng)
    for i in klist:
        ii = i - 1
        if change[ii] < 0:
            change[ii] = -min(-change[ii], out[ii] - 1)
        elif change[ii] > 0:
            change[ii] = min(change[ii], out[ii + 1] - 1)
        out[ii] += change[ii]
        out[ii + 1] -= change[ii]
    return out


def ch_dur_indep(
    sduri: np.ndarray,
    jprob: Union[int, float, np.ndarray],
    *,
    jfixed: bool = False,
    k: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Model ``indep``: independent timing shifts; spells may be erased."""
    rng = rng or np.random.RandomState()
    out = np.array(sduri, dtype=np.int64, copy=True)
    nt = _count_transitions(out)
    if nt <= 0:
        return out
    klist = _set_klist(k, nt, rng)
    change = np.zeros(nt, dtype=np.int64)
    if jfixed:
        change[klist - 1] = _move(1, jprob, rng)
    else:
        change[klist - 1] = _move(len(klist), jprob, rng)
    for i in klist:
        ii = i - 1
        if change[ii] < 0:
            chg = -change[ii]
            out[ii + 1] = out[ii + 1] + chg
            if out[ii] >= chg:
                out[ii] = out[ii] - chg
            else:
                rchg = chg - out[ii]
                out[ii] = 0
                jj = ii - 1
                while jj >= 0 and rchg > 0:
                    dur = out[jj]
                    out[jj] = max(0, dur - rchg)
                    rchg = rchg - (dur - out[jj])
                    jj -= 1
                if rchg > 0:
                    out[ii + 1] = out[ii + 1] - rchg
        else:
            chg = change[ii]
            out[ii] = out[ii] + chg
            if out[ii + 1] >= chg:
                out[ii + 1] = out[ii + 1] - chg
            else:
                rchg = chg - out[ii + 1]
                out[ii + 1] = 0
                jj = ii + 2
                while jj < nt + 1 and rchg > 0:
                    dur = out[jj]
                    out[jj] = max(0, dur - rchg)
                    rchg = rchg - (dur - out[jj])
                    jj += 1
                if rchg > 0:
                    out[ii] = out[ii] - rchg
    return out


def ch_dur_relat(
    sduri: np.ndarray,
    jprob: Union[int, float, np.ndarray],
    *,
    jfixed: bool = False,
    k: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Model ``relative``: preserve time to subsequent transitions after each shift."""
    rng = rng or np.random.RandomState()
    out = np.array(sduri, dtype=np.int64, copy=True)
    nt = _count_transitions(out)
    if nt <= 0:
        return out
    klist = _set_klist(k, nt, rng)
    change = np.zeros(nt, dtype=np.int64)
    if jfixed:
        change[klist - 1] = _move(1, jprob, rng)
    else:
        change[klist - 1] = _move(len(klist), jprob, rng)
    for i in klist:
        ii = i - 1
        if change[ii] < 0:
            chg = -change[ii]
            out[nt] = out[nt] + chg
            if out[ii] >= chg:
                out[ii] = out[ii] - chg
            else:
                rchg = chg - out[ii]
                out[ii] = 0
                jj = ii - 1
                while jj >= 0 and rchg > 0:
                    dur = out[jj]
                    out[jj] = max(0, dur - rchg)
                    rchg = rchg - (dur - out[jj])
                    jj -= 1
                if rchg > 0:
                    out[nt] = out[nt] - rchg
        else:
            chg = change[ii]
            out[ii] = out[ii] + chg
            if out[nt] >= chg:
                out[nt] = out[nt] - chg
            else:
                rchg = chg - out[nt]
                out[nt] = 0
                jj = nt - 1
                while jj > ii and rchg > 0:
                    dur = out[jj]
                    out[jj] = max(0, dur - rchg)
                    rchg = rchg - (dur - out[jj])
                    jj -= 1
                if rchg > 0:
                    out[ii] = out[ii] - rchg
    return out


def apply_timing_change(
    sdur_row: np.ndarray,
    *,
    ch_meth: int,
    jprob: Union[int, float, np.ndarray],
    jfixed: bool = False,
    kchanges: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Apply one of the three duration-change models to a single sequence."""
    if ch_meth == 1:
        return ch_dur(sdur_row, jprob, jfixed=jfixed, k=kchanges, rng=rng)
    if ch_meth == 2:
        return ch_dur_indep(sdur_row, jprob, jfixed=jfixed, k=kchanges, rng=rng)
    if ch_meth == 3:
        return ch_dur_relat(sdur_row, jprob, jfixed=jfixed, k=kchanges, rng=rng)
    raise ValueError(f"Unknown ch_meth code: {ch_meth}")
