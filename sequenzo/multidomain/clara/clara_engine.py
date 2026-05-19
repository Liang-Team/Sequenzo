"""
@Author  : Yuqi Liang 梁彧祺
@File    : clara_engine.py
@Time    : 18/05/2026 17:23
@Desc    : 
Provider-aware CLARA engine for multidomain sequence analysis.

This module mirrors :mod:`sequenzo.big_data.clara.clara` but delegates distance
computation to a :class:`~sequenzo.multidomain.clara.distance_providers.DistanceProvider`.
"""

from __future__ import annotations

import gc
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sequenzo.big_data.clara.clara import adjustedRandIndex, jaccardCoef
from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
from sequenzo.big_data.clara.utils.davies_bouldin import (
    davies_bouldin_internal,
    fuzzy_davies_bouldin_internal,
)
from sequenzo.big_data.clara.utils.get_weighted_diss import get_weighted_diss
from sequenzo.clustering.k_medoids import KMedoids
from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage
from sequenzo.clustering.sequences_to_variables.helske_regression_variables import (
    medoid_indices_from_kmedoids_result,
)
from sequenzo.define_sequence_data import SequenceData

from .distance_providers import DistanceProvider
from ._utils import check_sample_size_for_k


def _quality_from_distances(
    diss_to_medoids: np.ndarray,
    medoids: np.ndarray,
    *,
    method: str,
    weights: np.ndarray,
    m: float = 1.5,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Compute mean distance, DB, XB, PBM, AMS, and cluster labels from N x K distances.
    """
    alphabeta = np.array([np.sort(row)[:2] for row in diss_to_medoids])
    sil = (alphabeta[:, 1] - alphabeta[:, 0]) / np.maximum(alphabeta[:, 1], alphabeta[:, 0])

    if method == "fuzzy":
        mexp = -1.0 / (m - 1.0)
        memb = np.power(diss_to_medoids, mexp)
        zero_dist = diss_to_medoids == 0.0
        all_med = np.sum(zero_dist, axis=1) > 0
        memb[all_med, :] = 0.0
        memb[zero_dist] = 1.0
        memb = memb / memb.sum(axis=1, keepdims=True)
        mean_diss = float(np.sum(np.sum(np.power(memb, m) * diss_to_medoids, axis=1) * weights))
        db = fuzzy_davies_bouldin_internal(
            diss_to_medoids, memb, medoids, weights=weights
        )["db"]
        highest_memb = np.sort(memb, axis=1)[:, -2:]
        crispness = np.power(highest_memb[:, 1] - highest_memb[:, 0], 1.0)
        pbm = ((1 / len(medoids)) * (np.max(diss_to_medoids[medoids]) / mean_diss)) ** 2
        ams = float(np.sum(crispness * sil * weights) / np.sum(crispness * weights))
        labels = memb
    else:
        labels = np.argmin(diss_to_medoids, axis=1)
        mean_diss = float(np.sum(alphabeta[:, 0] * weights))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            db = davies_bouldin_internal(
                diss=diss_to_medoids,
                clustering=labels,
                medoids=medoids,
                weights=weights,
            )["db"]
        pbm = ((1 / len(medoids)) * (np.max(diss_to_medoids[medoids]) / mean_diss)) ** 2
        ams = float(np.sum(sil * weights))

    distmed = diss_to_medoids[medoids, :]
    distmed_flat = distmed[np.triu_indices_from(distmed, k=1)]
    minsep = float(np.min(distmed_flat)) if distmed_flat.size else 1.0
    xb = mean_diss / minsep if minsep > 0 else np.inf

    return mean_diss, db, pbm, ams, xb, labels


def _run_single_iteration(
    provider: DistanceProvider,
    *,
    sample_size: int,
    kvals: Sequence[int],
    method: str,
    aggregation: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """One CLARA iteration: sample, cluster the sample, assign all sequences."""
    n_agg = provider.n_sequences()
    probs = np.asarray(aggregation["probs"], dtype=float)
    max_k = max(kvals)

    # Without replacement when possible so unique sample size is not collapsed.
    replace = sample_size > n_agg
    sample_rows = rng.choice(
        n_agg,
        size=sample_size,
        p=probs,
        replace=replace,
    )
    sample_df = pd.DataFrame({"id": sample_rows})
    ac2 = DataFrameAggregator().aggregate(sample_df)
    local_indices = sample_rows[np.asarray(ac2["aggIndex"], dtype=int) - 1]
    n_unique_sample = len(local_indices)
    if n_unique_sample < max_k:
        raise ValueError(
            f"Only {n_unique_sample} unique sampled cases after aggregation, but "
            f"max(kvals)={max_k}. Increase sample_size, reduce k, or set "
            f"sample_size <= n_unique_cases ({n_agg}) to sample without replacement."
        )

    diss_sample = provider.sample_distance_matrix(local_indices)
    diss_sample = np.asarray(diss_sample, dtype=float)
    weighted = get_weighted_diss(diss_sample.copy(), ac2["aggWeights"])
    hc = linkage(weighted, method="ward")

    outputs: List[Dict[str, Any]] = []
    if method != "crisp":
        raise ValueError("Only method='crisp' is supported in this CLARA engine version.")

    for k in kvals:
        clustering = KMedoids(
            diss=diss_sample,
            k=k,
            initialclust=hc,
            weights=ac2["aggWeights"],
            verbose=False,
        )
        medoid_rows = medoid_indices_from_kmedoids_result(clustering)
        medoids = local_indices[medoid_rows]

        diss_full = provider.distance_to_medoids(medoids)
        mean_diss, db, pbm, ams, xb, labels = _quality_from_distances(
            diss_full,
            medoids,
            method="crisp",
            weights=probs,
        )

        outputs.append(
            {
                "mean_diss": mean_diss,
                "db": db,
                "pbm": pbm,
                "ams": ams,
                "xb": xb,
                "clustering": labels,
                "medoids": medoids,
            }
        )

    del diss_sample
    gc.collect()
    return outputs


def clara_from_distance_provider(
    provider: DistanceProvider,
    *,
    reference_seqdata: SequenceData,
    aggregation: Optional[Dict[str, Any]] = None,
    R: int = 100,
    sample_size: Optional[int] = None,
    kvals: Optional[Sequence[int]] = None,
    method: str = "crisp",
    criteria: Sequence[str] = ("distance",),
    stability: bool = False,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run CLARA using a distance provider instead of a full distance matrix.

    Parameters
    ----------
    provider
        Object implementing sample and medoid-distance queries.
    reference_seqdata
        Sequence object used for case aggregation and ID alignment (typically
        the first domain before aggregation).
    """
    if kvals is None:
        kvals = list(range(2, 11))
    else:
        kvals = list(kvals)

    if sample_size is None:
        sample_size = 40 + 2 * max(kvals)

    if aggregation is None:
        ac = DataFrameAggregator().aggregate(reference_seqdata.seqdata)
    else:
        ac = dict(aggregation)

    check_sample_size_for_k(
        sample_size,
        kvals,
        n_unique_cases=len(ac["aggWeights"]),
    )

    method = method.lower()
    if method != "crisp":
        raise ValueError(
            "clara_from_distance_provider currently supports method='crisp' only. "
            "Fuzzy and representativeness will be added in a later release."
        )

    criteria = tuple(c.lower() for c in criteria)
    if len(criteria) != 1:
        raise ValueError(
            "Exactly one clustering criterion is supported per run. "
            f"Got {criteria!r}. Use a single value such as criteria=('distance',). "
            "Multi-criterion output (result.by_criterion) is planned for a later release."
        )
    criterion = criteria[0]
    valid = {"distance", "db", "xb", "pbm", "ams"}
    if criterion not in valid:
        raise ValueError(f"criterion must be one of {sorted(valid)}; got {criterion!r}.")

    if verbose:
        print("[>] Starting multidomain CLARA with distance provider.")
        print(f"  - Strategy sample size: {sample_size}, iterations: {R}")

    ac["probs"] = np.asarray(ac["aggWeights"], dtype=float) / len(reference_seqdata.seqdata)

    rng = np.random.default_rng(random_state)
    iteration_seeds = rng.integers(0, np.iinfo(np.int64).max, size=R)

    def _iteration(seed: int) -> List[Dict[str, Any]]:
        iter_rng = np.random.default_rng(seed)
        return _run_single_iteration(
            provider,
            sample_size=sample_size,
            kvals=kvals,
            method="crisp",
            aggregation=ac,
            rng=iter_rng,
        )

    if verbose:
        print("[>] Running CLARA iterations...")

    if n_jobs == 1:
        results = [_iteration(int(iteration_seeds[i])) for i in range(R)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_iteration)(int(iteration_seeds[i])) for i in range(R)
        )

    if verbose:
        print("  - Done.")
        print("[>] Aggregating iterations for each k...")

    collected: List[List[Dict[str, Any]]] = [[] for _ in kvals]
    for iter_result in results:
        for k_idx, item in enumerate(iter_result):
            collected[k_idx].append(item)

    kret: List[Dict[str, Any]] = []

    for k_index, k_value in enumerate(kvals):
        bucket = collected[k_index]
        mean_all = [d["mean_diss"] for d in bucket]
        db_all = [d["db"] for d in bucket]
        pbm_all = [d["pbm"] for d in bucket]
        ams_all = [d["ams"] for d in bucket]
        xb_all = [d["xb"] for d in bucket]
        clustering_all = [d["clustering"] for d in bucket]
        med_all = [d["medoids"] for d in bucket]

        objective_map = {
            "distance": mean_all,
            "pbm": pbm_all,
            "db": db_all,
            "ams": ams_all,
            "xb": xb_all,
        }
        objective = objective_map[criterion]
        best = int(
            np.argmax(objective) if criterion in {"ams", "pbm"} else np.argmin(objective)
        )

        if stability:

            def _stability_pair(j: int) -> Tuple[float, float]:
                left_labels = clustering_all[j]
                right_labels = clustering_all[best]
                df = pd.DataFrame(
                    {
                        "left": left_labels,
                        "right": right_labels,
                        "w": ac["aggWeights"],
                    }
                )
                tab = df.groupby(["left", "right"])["w"].sum().unstack(fill_value=0)
                return adjustedRandIndex(tab), jaccardCoef(tab)

            if n_jobs == 1:
                arilist = [_stability_pair(j) for j in range(R)]
            else:
                arilist = Parallel(n_jobs=n_jobs)(
                    delayed(_stability_pair)(j) for j in range(R)
                )
            arimatrix = pd.DataFrame(arilist, columns=["ARI", "JC"])
            ari08 = int(np.sum(arimatrix.iloc[:, 0] >= 0.8))
            jc08 = int(np.sum(arimatrix.iloc[:, 1] >= 0.8))
            stability_info = {
                "ari": arimatrix["ARI"].to_numpy(),
                "jc": arimatrix["JC"].to_numpy(),
                "ari08": ari08,
                "jc08": jc08,
                "mean_ari": float(arimatrix["ARI"].mean()),
                "mean_jc": float(arimatrix["JC"].mean()),
                "trimmed_mean_ari": float(
                    np.mean(np.sort(arimatrix["ARI"].to_numpy())[-max(1, R // 5) :])
                ),
                "trimmed_mean_jc": float(
                    np.mean(np.sort(arimatrix["JC"].to_numpy())[-max(1, R // 5) :])
                ),
            }
        else:
            arimatrix = None
            ari08 = np.nan
            jc08 = np.nan
            stability_info = None

        best_clustering = clustering_all[best]
        disag = np.full(reference_seqdata.seqdata.shape[0], -1, dtype=float)
        for row_idx, agg_idx in enumerate(np.asarray(ac["disaggIndex"], dtype=int) - 1):
            disag[row_idx] = best_clustering[agg_idx] + 1

        evol = (
            np.maximum.accumulate(objective)
            if criterion in {"ams", "pbm"}
            else np.minimum.accumulate(objective)
        )

        bestcluster = {
            "medoids": np.asarray(ac["aggIndex"], dtype=int)[med_all[best]] - 1,
            "medoids_agg": med_all[best],
            "clustering": disag,
            "membership": None,
            "evol_diss": evol,
            "iter_objective": objective,
            "objective": objective[best],
            "iteration": best,
            "arimatrix": arimatrix,
            "stability": stability_info,
            "criteria": criterion,
            "method": method,
            "avg_dist": mean_all[best],
            "pbm": pbm_all[best],
            "db": db_all[best],
            "xb": xb_all[best],
            "ams": ams_all[best],
            "ari08": ari08,
            "jc08": jc08,
            "R": R,
            "k": k_value,
            "sample_size": sample_size,
        }

        kret.append(
            {
                "k": k_value,
                "criteria": criterion,
                "stats": [
                    bestcluster["avg_dist"],
                    bestcluster["pbm"],
                    bestcluster["db"],
                    bestcluster["xb"],
                    bestcluster["ams"],
                    bestcluster["ari08"],
                    bestcluster["jc08"],
                    best,
                ],
                "bestcluster": bestcluster,
            }
        )

    return _package_clara_output(
        kret,
        kvals=kvals,
        criterion=criterion,
        method=method,
        stability=stability,
        reference_seqdata=reference_seqdata,
        sample_size=sample_size,
        R=R,
    )


def _package_clara_output(
    kret: List[Dict[str, Any]],
    *,
    kvals: Sequence[int],
    criterion: str,
    method: str,
    stability: bool,
    reference_seqdata: SequenceData,
    sample_size: int,
    R: int,
) -> Dict[str, Any]:
    """Build clustering/stats structures for a single criterion."""
    clustering = pd.DataFrame(
        -1.0,
        index=reference_seqdata.ids,
        columns=[f"Cluster {k}" for k in kvals],
    )
    stats = np.full((len(kvals), 8), -1.0, dtype=float)
    clara_dict: Dict[int, Any] = {}

    for entry in kret:
        k = entry["k"]
        pos = kvals.index(k)
        stats[pos, :] = np.array(entry["stats"], dtype=float)
        clara_dict[k] = entry["bestcluster"]
        clustering.iloc[:, pos] = entry["bestcluster"]["clustering"]

    stats_df = pd.DataFrame(
        stats,
        columns=["avg_dist", "pbm", "db", "xb", "ams", "ari08", "jc08", "best_iter"],
    )
    stats_df.insert(0, "k", list(kvals))
    stats_df["criterion"] = criterion

    return {
        "kvals": list(kvals),
        "clara": clara_dict,
        "clustering": clustering,
        "stats": stats_df,
        "stability": stability,
        "method": method,
        "sample_size": sample_size,
        "R": R,
        "criterion": criterion,
    }


__all__ = ["clara_from_distance_provider"]
