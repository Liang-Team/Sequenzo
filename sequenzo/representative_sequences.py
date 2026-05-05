"""
@Author  : 梁彧祺 Yuqi Liang
@File    : representative_sequences.py
@Time    : 05/05/2026 09:17
@Desc    :
    Representative sequence/object utilities aligned with TraMineR.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import _to_square_matrix
from sequenzo.visualization.plot_relative_frequency import _cmdscale


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / wsum)


def get_distance_center(
    diss: np.ndarray,
    group: np.ndarray | None = None,
    medoids_index: str | bool | None = None,
    allcenter: bool = False,
    weights: np.ndarray | None = None,
    squared: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """
    Equivalent to TraMineR::disscenter().
    """
    if allcenter:
        raise NotImplementedError("`allcenter=True` is not implemented yet.")

    D = _to_square_matrix(diss).astype(float)
    if squared:
        D = D**2
    n = D.shape[0]
    if group is None:
        group = np.ones(n, dtype=int)
    else:
        group = np.asarray(group)
        if group.shape[0] != n:
            raise ValueError("Length of `group` must equal nrow(diss).")
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != n:
            raise ValueError("Length of `weights` must equal nrow(diss).")

    if isinstance(medoids_index, bool):
        medoids_index = "first" if medoids_index else None
    if medoids_index is not None:
        medoids_index = str(medoids_index).lower()
        if medoids_index not in {"first", "all"}:
            raise ValueError("`medoids_index` should be one of 'first', 'all', or None.")

    out = np.zeros(n, dtype=float)
    medoids: list[np.ndarray] | list[int] = []
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        sub = D[np.ix_(idx, idx)]
        w = weights[idx]
        dc = sub @ w
        dc = dc - _weighted_mean(dc, w) / 2.0
        out[idx] = dc
        if medoids_index is not None:
            mind = np.min(dc)
            candidates = idx[np.where(np.isclose(dc, mind))[0]]
            if medoids_index == "all":
                medoids.append(candidates)
            else:
                medoids.append(int(np.sort(candidates)[0]))

    if medoids_index is None:
        return out
    if medoids_index == "all":
        if len(medoids) == 1:
            return medoids[0]
        return medoids
    return np.asarray(medoids, dtype=int)


def get_relative_frequency_groups(
    diss: np.ndarray,
    k: int | None = None,
    sortv: str | np.ndarray | None = "mds",
    weights: np.ndarray | None = None,
    grp_meth: str = "prop",
    squared: bool = False,
    pow: float | None = None,
) -> dict[str, Any]:
    """
    Equivalent to TraMineR::dissrf().
    """
    D = _to_square_matrix(diss).astype(float)
    ncase = D.shape[0]
    if weights is None:
        weights = np.ones(ncase, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != ncase:
            raise ValueError("Length of `weights` must equal nrow(diss).")
        if grp_meth != "prop":
            raise ValueError("Weighted data requires grp_meth='prop'.")
        weights = ncase * weights / np.sum(weights)

    wsum = float(np.sum(weights))
    if k is None:
        k = int(min(np.floor(wsum / 10.0), 100))
        k = max(1, k)
    if pow is None:
        pow = 2.0 if squared else 1.0
    mdspow = 2 if squared else 1

    gmedoid = int(get_distance_center(D, medoids_index="first", weights=weights, squared=squared)[0])
    gmedoid_dist = D[:, gmedoid]
    sum_gmedoid_dist = float(np.sum(gmedoid_dist**pow))

    if sortv is None:
        sort_values = np.arange(ncase, dtype=float)
    elif isinstance(sortv, str):
        if sortv != "mds":
            raise ValueError("If string, `sortv` must be 'mds'.")
        sort_values = _cmdscale(D**mdspow)[:, 0]
    else:
        sort_values = np.asarray(sortv, dtype=float).reshape(-1)
        if sort_values.shape[0] != ncase:
            raise ValueError("Length of `sortv` must equal nrow(diss).")

    sortorder = np.argsort(sort_values)
    cumweights = np.cumsum(weights[sortorder])

    if grp_meth not in {"prop", "first"}:
        raise ValueError("`grp_meth` must be one of {'prop','first'}.")

    index_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    medoids = np.zeros(k, dtype=int)

    if grp_meth == "prop":
        gsize = wsum / k
        sumwdist = np.zeros(k, dtype=float)
        start = 0
        for i in range(k):
            if i == k - 1:
                end = ncase
            else:
                end = int(np.searchsorted(cumweights, (i + 1) * gsize, side="left") + 1)
                end = min(max(end, start + 1), ncase)
            ind = sortorder[start:end]
            start = end
            index_list.append(ind)
            w = weights[ind].copy()
            weights_list.append(w)
            if ind.size == 1:
                medoids[i] = int(ind[0])
                d = np.array([0.0])
            else:
                dd = D[np.ix_(ind, ind)]
                m_local = int(get_distance_center(dd, medoids_index="first", weights=w, squared=squared)[0])
                medoids[i] = int(ind[m_local])
                d = dd[:, m_local]
            dist_list.append(d)
            sumwdist[i] = float(np.sum(w * (d**pow)))
        r2 = 1.0 - (float(np.sum(sumwdist)) / sum_gmedoid_dist if sum_gmedoid_dist > 0 else 0.0)
    else:
        ng = int(np.floor(wsum / k))
        r = int(wsum % k)
        n_per_group = np.repeat(ng, k)
        if r > 0:
            n_per_group[:r] += 1
        labels = np.zeros(ncase, dtype=int)
        labels[sortorder] = np.repeat(np.arange(k), n_per_group)[:ncase]
        kmedoid_dist = np.zeros(ncase, dtype=float)
        for i in range(k):
            ind = np.where(labels == i)[0]
            index_list.append(ind)
            weights_list.append(np.ones(ind.size, dtype=float))
            if ind.size == 1:
                medoids[i] = int(ind[0])
                d = np.array([0.0])
            else:
                dd = D[np.ix_(ind, ind)]
                m_local = int(get_distance_center(dd, medoids_index="first", squared=squared)[0])
                medoids[i] = int(ind[m_local])
                d = dd[:, m_local]
                kmedoid_dist[ind] = d
            dist_list.append(d)
        r2 = 1.0 - (float(np.sum(kmedoid_dist**pow)) / sum_gmedoid_dist if sum_gmedoid_dist > 0 else 0.0)

    esd = r2 / (k - 1) if k > 1 else 0.0
    usd = (1.0 - r2) / (wsum - k) if wsum > k else np.nan
    fstat = float(esd / usd) if (usd is not None and np.isfinite(usd) and usd > 0) else np.nan

    return {
        "medoids": medoids,
        "dist_list": dist_list,
        "index_list": index_list,
        "weights_list": weights_list,
        "R2": float(r2),
        "Fstat": fstat,
        "pvalue": np.nan,
        "grp_meth": grp_meth,
    }


def get_relative_frequency_representatives(
    seqdata: SequenceData,
    diss: np.ndarray,
    k: int | None = None,
    sortv: str | np.ndarray | None = "mds",
    weights: np.ndarray | None = None,
    weighted: bool = True,
    grp_meth: str = "prop",
    squared: bool = False,
    pow: float | None = None,
) -> dict[str, Any]:
    """
    Equivalent to TraMineR::seqrf().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    if weights is None and weighted:
        weights = getattr(seqdata, "weights", None)
    if not weighted:
        weights = None
    rf = get_relative_frequency_groups(
        diss, k=k, sortv=sortv, weights=weights, grp_meth=grp_meth, squared=squared, pow=pow
    )
    seqtoplot = seqdata.values[rf["medoids"]]
    return {"seqtoplot": seqtoplot, "rf": rf}


def get_representative_objects(
    diss: np.ndarray,
    criterion: str = "density",
    score: np.ndarray | None = None,
    decreasing: bool = True,
    coverage: float = 0.25,
    nrep: int | None = None,
    pradius: float = 0.10,
    dmax: float | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Equivalent to TraMineR::dissrep().
    """
    D = _to_square_matrix(diss).astype(float)
    nbobj = D.shape[0]
    if weights is None:
        weights = np.ones(nbobj, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != nbobj:
            raise ValueError("Number of weights must equal number of objects.")
    weights_sum = float(np.sum(weights))
    if dmax is None:
        dmax = float(np.max(D))
    if not (0.0 <= pradius <= 1.0):
        raise ValueError("pradius must be between 0 and 1.")
    radius = dmax * pradius

    if score is None:
        if criterion == "density":
            neighbours = D < radius
            score = neighbours @ weights
            decreasing = True
        elif criterion == "freq":
            neighbours = D == 0
            score = neighbours @ weights
            decreasing = True
        elif criterion == "dist":
            score = D @ weights
            decreasing = False
        elif criterion == "random":
            score = np.random.permutation(np.arange(nbobj))
            decreasing = False
        else:
            raise ValueError("Unknown criterion / no score provided.")
    score = np.asarray(score)
    if score.shape[0] != nbobj:
        raise ValueError("Score must be a vector of length equal to number of objects.")

    score_sort = np.argsort(score)
    if decreasing:
        score_sort = score_sort[::-1]
    rep_dist = D[np.ix_(score_sort, score_sort)]

    idx = 0
    idxrep: list[int] = []
    if nrep is None and coverage > 0:
        pctrep = 0.0
        while pctrep < coverage and idx < nbobj:
            idx += 1
            pos = idx - 1
            if pos == 0 or all(rep_dist[pos, np.array(idxrep)] > radius):
                idxrep.append(pos)
                tempm = rep_dist[:, np.array(idxrep)]
                nbnear = float(np.sum((np.sum(tempm < radius, axis=1) > 0) * weights[score_sort]))
                pctrep = nbnear / weights_sum if weights_sum > 0 else 0.0
    else:
        target = int(nrep if nrep is not None else 0)
        repcount = 0
        while repcount < target and idx < nbobj:
            idx += 1
            pos = idx - 1
            if pos == 0 or all(rep_dist[pos, np.array(idxrep)] > radius):
                idxrep.append(pos)
                repcount += 1

    picked = score_sort[np.array(idxrep, dtype=int)]
    dist_repseq = D[:, picked]
    dc_tot = get_distance_center(D, weights=weights)
    dist_to_rep = np.min(dist_repseq, axis=1)
    min_idx = np.argmin(dist_repseq, axis=1)
    quality = (
        (float(np.sum(dc_tot * weights)) - float(np.sum(dist_to_rep * weights))) / float(np.sum(dc_tot * weights))
        if np.sum(dc_tot * weights) > 0
        else 0.0
    )
    return {
        "indices": picked,
        "scores": score,
        "distances": dist_repseq,
        "rep_group": min_idx + 1,
        "quality": float(quality),
        "criterion": criterion,
    }


def get_representative_sequences(
    seqdata: SequenceData,
    criterion: str = "density",
    score: np.ndarray | None = None,
    decreasing: bool = True,
    coverage: float = 0.25,
    nrep: int | None = None,
    pradius: float = 0.10,
    dmax: float | None = None,
    diss: np.ndarray | None = None,
    weighted: bool = True,
) -> dict[str, Any]:
    """
    Equivalent to TraMineR::seqrep().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    if diss is None:
        raise ValueError("`diss` must be provided (distance matrix).")
    weights = getattr(seqdata, "weights", None)
    if (not weighted) or (weights is None):
        weights = np.ones(seqdata.n_sequences, dtype=float)
    rep = get_representative_objects(
        diss=diss,
        criterion=criterion,
        score=score,
        decreasing=decreasing,
        coverage=coverage,
        nrep=nrep,
        pradius=pradius,
        dmax=dmax,
        weights=weights,
    )
    return {
        "indices": rep["indices"],
        "sequences": seqdata.values[rep["indices"]],
        "scores": rep["scores"],
        "distances": rep["distances"],
        "quality": rep["quality"],
    }
