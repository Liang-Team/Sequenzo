"""
@Author  : 李欣怡 Xinyi Li
@File    : clara.py
@Time    : 2024/12/27 12:04
@Desc    : 
"""

import gc
import os
from contextlib import redirect_stdout
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# from Tutorials.test import result
from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage
from scipy.special import comb
from itertools import product

from sequenzo.big_data.clara.utils.aggregatecases import *
from sequenzo.big_data.clara.utils.davies_bouldin import *
from sequenzo.big_data.clara.utils.get_weighted_diss import get_weighted_diss
from scipy.cluster.hierarchy import cut_tree

from sequenzo.clustering.fuzzy_clustering import wfcmdd
from sequenzo.clustering.k_medoids import KMedoids
from sequenzo.clustering.sequences_to_variables.helske_regression_variables import (
    fanny_membership,
    medoid_indices_from_kmedoids_result,
)

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix


def adjustedRandIndex(x, y=None):
    if isinstance(x, np.ndarray):
        x = np.array(x)
        y = np.array(y)
        if len(x) != len(y):
            raise ValueError("Arguments must be vectors of the same length")

        tab = pd.crosstab(x, y)
    else:
        tab = x

    if tab.shape == (1, 1):
        return 1

    # 计算 ARI 的四个部分：a, b, c, d
    a = np.sum(comb(tab.to_numpy(), 2))  # 选择每对组合的组合数
    b = np.sum(comb(np.sum(tab.to_numpy(), axis=1), 2)) - a
    c = np.sum(comb(np.sum(tab.to_numpy(), axis=0), 2)) - a
    d = comb(np.sum(tab.to_numpy()), 2) - a - b - c

    ARI = (a - (a + b) * (a + c) / (a + b + c + d)) / ((a + b + a + c) / 2 - (a + b) * (a + c) / (a + b + c + d))
    return ARI


def jaccardCoef(tab):
    if tab.shape == (1, 1):
        return 1

    # 计算交集（n11）和并集（n01 和 n10）
    n11 = np.sum(tab.to_numpy() ** 2)  # 交集
    n01 = np.sum(np.sum(tab.to_numpy(), axis=0) ** 2)  # 列和的平方
    n10 = np.sum(np.sum(tab.to_numpy(), axis=1) ** 2)  # 行和的平方

    return n11 / (n01 + n10 - n11)


def _membership_from_labels(labels: np.ndarray) -> np.ndarray:
    """Convert a hard partition to a binary membership matrix."""
    labels = np.asarray(labels).reshape(-1)
    unique_labels = np.unique(labels)
    membership = np.zeros((labels.size, unique_labels.size), dtype=float)
    for cluster_idx, cluster_id in enumerate(unique_labels):
        membership[labels == cluster_id, cluster_idx] = 1.0
    return membership


def _optional_fanny_seed_unavailable(exc: ValueError) -> bool:
    """Return True when optional FANNY seeding cannot run for valid CLARA data."""
    message = str(exc)
    return (
        "k must be at most n//2 - 1" in message
        or "all-zero distances cannot define multiple fuzzy clusters" in message
        or "For R cluster::fanny parity" in message
    )


def clara(seqdata, R=100, kvals=None, sample_size=None, method="crisp", dist_args=None,
          criteria=["distance"], stability=False, max_dist=None, m=1.5, dnoise=None):

    # ==================
    # Parameter checking
    # ==================
    if kvals is None:
        kvals = range(2, 11)

    if sample_size is None:
        sample_size = 40 + 2 * max(kvals)

    print("[>] Starting generalized CLARA for sequence analysis.")

    # Check for input data type (should be a sequence object)
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] 'seqdata' should be SequenceData, check the input format.")

    if max(kvals) > sample_size:
        raise ValueError("[!] More clusters than the size of the sample requested.")

    allmethods = ["crisp", "fuzzy", "representativeness", "noise"]
    method = method.lower()
    if method not in allmethods:
        raise ValueError(f"[!] Unknown method {method}. Please specify one of the following: {', '.join(allmethods)}")

    if method == "representativeness":
        if max_dist is None:
            raise ValueError("[!] You need to set max.dist when using representativeness method.")
        if isinstance(max_dist, (bool, np.bool_)):
            raise ValueError("[!] max.dist must be a positive finite number.")
        try:
            max_dist = float(max_dist)
        except (TypeError, ValueError) as exc:
            raise ValueError("[!] max.dist must be a positive finite number.") from exc
        if not np.isfinite(max_dist) or max_dist <= 0:
            raise ValueError("[!] max.dist must be a positive finite number.")

    allcriteria = ["distance", "db", "xb", "pbm", "ams"]
    criteria = [c.lower() for c in criteria]
    if not all(c in allcriteria for c in criteria):
        raise ValueError(
            f"[!] Unknown criteria among {', '.join(criteria)}. Please specify at least one among {', '.join(allcriteria)}.")

    if dist_args is None:
        raise ValueError("[!] You need to set the 'dist_args' for get_distance_matrix function.")

    print(f"[>] Using {method} clustering optimizing the following criterion: {', '.join(criteria)}.")

    # FIXME : Add coherance check between method and criteria

    # ===========
    # Aggregation
    # ===========
    number_seq = len(seqdata.seqdata)
    print(f"  - Aggregating {number_seq} sequences...")

    ac = DataFrameAggregator().aggregate(seqdata.seqdata)
    agseqdata = seqdata.seqdata.iloc[np.asarray(ac["aggIndex"], dtype=int) - 1, :]
    # agseqdata.attrs['weights'] = None
    ac['probs'] = ac['aggWeights'] / number_seq
    print(f"  - OK ({len(ac['aggWeights'])} unique cases).")
    if len(agseqdata) < max(kvals):
        raise ValueError("[!] Fewer unique cases than requested clusters after aggregation.")

    # Memory cleanup before parallel computation
    gc.collect()
    print("[>] Starting iterations...")

    def calc_pam_iter(circle, agseqdata, sample_size, kvals, ac):
        # Sampling with replacement allows the process to proceed normally
        # even when the sample size exceeds the dataset size, as samples can be repeatedly drawn."
        n_unique = len(agseqdata)
        probs = np.asarray(ac["probs"], dtype=float)
        min_unique = min(max(kvals), n_unique)
        if sample_size >= n_unique:
            required = np.arange(n_unique, dtype=int)
            extra = np.random.choice(
                n_unique, size=sample_size - n_unique, p=probs, replace=True
            )
            mysample = np.concatenate([required, extra])
        else:
            mysample = np.random.choice(n_unique, size=sample_size, p=probs, replace=True)
            if np.unique(mysample).size < min_unique:
                required = np.random.choice(n_unique, size=min_unique, p=probs, replace=False)
                mysample[:min_unique] = required
        mysample = pd.DataFrame({'id': mysample})

        # Re-aggregate!
        ac2 = DataFrameAggregator().aggregate(mysample)
        data_subset = agseqdata.iloc[mysample.iloc[np.asarray(ac2["aggIndex"], dtype=int) - 1, 0], :]

        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                states = np.arange(1, len(seqdata.states) + 1).tolist()
                data_subset = SequenceData(data_subset,
                                           time=seqdata.time,
                                           states=states)
                dist_args['seqdata'] = data_subset
                diss = get_distance_matrix(opts=dist_args)

        diss = diss.values
        _diss = diss.copy()
        _diss = get_weighted_diss(_diss, ac2['aggWeights'])
        hc = linkage(_diss, method='ward')
        del _diss

        # For each number of clusters
        allclust = []

        for k in kvals:
            iter_dnoise = dnoise
            if method in ("fuzzy", "noise"):
                seed_labels = cut_tree(hc, n_clusters=k).ravel()
                seed_membership = _membership_from_labels(seed_labels)
                algo = "FCMdd" if method == "fuzzy" else "NCdd"
                clustering_c = wfcmdd(
                    diss,
                    memb=seed_membership,
                    weights=ac2["aggWeights"],
                    method=algo,
                    m=m,
                    dnoise=iter_dnoise,
                )
                clustering = clustering_c
                try:
                    fanny_membership_matrix, _ = fanny_membership(
                        diss,
                        k=k,
                        m=m,
                        weights=ac2["aggWeights"],
                    )
                except ValueError as exc:
                    if not _optional_fanny_seed_unavailable(exc):
                        raise
                else:
                    clustering_f = wfcmdd(
                        diss,
                        memb=fanny_membership_matrix,
                        weights=ac2["aggWeights"],
                        method=algo,
                        m=m,
                        dnoise=iter_dnoise,
                    )
                    if clustering_f.functional < clustering.functional:
                        clustering = clustering_f
                if method == "noise":
                    iter_dnoise = clustering.dnoise
                medoids = mysample.iloc[np.asarray(ac2["aggIndex"], dtype=int)[clustering.mobile_centers] - 1].to_numpy().flatten()
            else:
                clustering = KMedoids(
                    diss=diss,
                    k=k,
                    initialclust=hc,
                    weights=ac2["aggWeights"],
                    verbose=False,
                )
                medoid_rows = medoid_indices_from_kmedoids_result(clustering)
                medoids = mysample.iloc[np.asarray(ac2["aggIndex"], dtype=int)[medoid_rows] - 1].to_numpy().flatten()

            refseq = [list(range(0, len(agseqdata))), medoids.tolist()]
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull):
                    states = np.arange(1, len(seqdata.states) + 1).tolist()
                    agseqdata_wrapped = SequenceData(
                        agseqdata,
                        time=seqdata.time,
                        states=states,
                    )
                    dist_args["seqdata"] = agseqdata_wrapped
                    dist_args["refseq"] = refseq
                    diss2 = get_distance_matrix(opts=dist_args)
                    del dist_args["refseq"]

            diss2 = diss2.to_numpy()
            alphabeta = np.array([np.sort(row)[:2] for row in diss2])
            sil = (alphabeta[:, 1] - alphabeta[:, 0]) / np.maximum(alphabeta[:, 1], alphabeta[:, 0])

            if method == "fuzzy":
                mexp = -1.0 / (m - 1.0)
                memb = np.power(diss2, mexp)
                zero_dist = diss2 == 0.0
                all_med = np.sum(zero_dist, axis=1) > 0
                memb[all_med, :] = 0.0
                memb[zero_dist] = 1.0
                memb = memb / memb.sum(axis=1, keepdims=True)
                mean_diss = np.sum(np.sum(np.power(memb, m) * diss2, axis=1) * ac["probs"])
                db = fuzzy_davies_bouldin_internal(diss2, memb, medoids, weights=ac["aggWeights"])["db"]
                highest_memb = np.sort(memb, axis=1)[:, -2:]
                crispness = np.power(highest_memb[:, 1] - highest_memb[:, 0], 1.0)
                pbm = ((1 / len(medoids)) * (np.max(diss2[medoids]) / mean_diss)) ** 2
                ams = np.sum(crispness * sil * ac["probs"]) / np.sum(crispness * ac["probs"])
            elif method == "noise":
                diss3 = np.column_stack([diss2, np.full(diss2.shape[0], iter_dnoise)])
                mexp = -1.0 / (m - 1.0)
                memb = np.power(diss3, mexp)
                zero_dist = diss3 == 0.0
                all_med = np.sum(zero_dist, axis=1) > 0
                memb[all_med, :] = 0.0
                memb[zero_dist] = 1.0
                memb = memb / memb.sum(axis=1, keepdims=True)
                mean_diss = np.sum(np.sum(np.power(memb, m) * diss3, axis=1) * ac["probs"])
                db = fuzzy_davies_bouldin_internal(
                    diss2,
                    memb[:, :-1],
                    medoids,
                    weights=ac["aggWeights"],
                )["db"]
                highest_memb = np.sort(memb[:, :-1], axis=1)[:, -2:]
                crispness = np.power(highest_memb[:, 1] - highest_memb[:, 0], 1.0)
                pbm = ((1 / len(medoids)) * (np.max(diss2[medoids]) / mean_diss)) ** 2
                ams = np.sum(crispness * sil * ac["probs"]) / np.sum(crispness * ac["probs"])
            else:
                memb = np.argmin(diss2, axis=1)
                mean_diss = np.sum(alphabeta[:, 0] * ac["probs"])
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                db = davies_bouldin_internal(
                    diss=diss2,
                    clustering=memb,
                    medoids=medoids,
                    weights=ac["aggWeights"],
                )["db"]
                warnings.resetwarnings()
                pbm = ((1 / len(medoids)) * (np.max(diss2[medoids]) / mean_diss)) ** 2
                ams = np.sum(sil * ac["probs"])

            distmed = diss2[medoids, :]
            distmed_flat = distmed[np.triu_indices_from(distmed, k=1)]
            minsep = np.min(distmed_flat)
            xb = mean_diss / minsep

            allclust.append(
                {
                    "mean_diss": mean_diss,
                    "db": db,
                    "pbm": pbm,
                    "ams": ams,
                    "xb": xb,
                    "clustering": memb,
                    "medoids": medoids,
                }
            )

        del diss
        gc.collect()

        return allclust

    # Compute in parallel using joblib
    # the output example of `results`:
    #         results[0] = all iter1's = [{k=2's}, {k=3's}, ... , {k=10's}]
    #         results[1] = all iter2's = [{k=2's}, {k=3's}, ... , {k=10's}]
    results = Parallel(n_jobs=-1)(
        delayed(calc_pam_iter)(circle=i, agseqdata=agseqdata, sample_size=sample_size, kvals=kvals, ac=ac) for i in range(R))
    # results = []
    # for i in range(R):
    #     res = calc_pam_iter(circle=i,
    #                         agseqdata=agseqdata,
    #                         sample_size=sample_size,
    #                         kvals=kvals,
    #                         ac=ac)
    #     results.append(res)

    print("  - Done.")
    print("[>] Aggregating iterations for each k values...")

    # aggregated output example :
    #         data[0] = all k=2's = [{when iter1, k=2's}, {when iter2, k=2's}, ... , {when iter100, k=2's}]
    #         data[1] = all k=3's = [{when iter1, k=3's}, {when iter2, k=3's}, ... , {when iter100, k=3's}]
    collected_data = [[] for _ in kvals]
    for iter_result in results:
        k = 0
        for item in iter_result:
            collected_data[k].append(item)
            k += 1

    kvalscriteria = list(product(range(len(kvals)), criteria))
    kret = []
    for item in kvalscriteria:
        k_index = item[0]
        k_value = kvals[k_index]
        _criteria = item[1]

        mean_all_diss = [d['mean_diss'] for d in collected_data[k_index]]
        db_all = [d['db'] for d in collected_data[k_index]]
        pbm_all = [d['pbm'] for d in collected_data[k_index]]
        ams_all = [d['ams'] for d in collected_data[k_index]]
        xb_all = [d['xb'] for d in collected_data[k_index]]
        clustering_all_diss = [d['clustering'] for d in collected_data[k_index]]
        med_all_diss = [d['medoids'] for d in collected_data[k_index]]

        # Find best clustering
        objective = {
            "distance": mean_all_diss,
            "pbm": pbm_all,
            "db": db_all,
            "ams": ams_all,
            "xb": xb_all
        }
        objective = objective[_criteria]
        best = np.argmax(objective) if _criteria in ["ams", "pbm"] else np.argmin(objective)

        # Compute clustering stability of the best partition
        if stability:
            def stability_labels(clustering):
                # ARI/JC stability is defined on crisp partitions here. Fuzzy/noise
                # memberships are reduced only for this stability cross-tab.
                clustering = np.asarray(clustering)
                if clustering.ndim == 1:
                    return clustering
                if clustering.ndim == 2:
                    return np.argmax(clustering, axis=1)
                raise ValueError("clustering must be a vector or membership matrix")

            def process_task(j, clustering_all_diss, ac, best):
                df = pd.DataFrame({
                    'clustering_j': stability_labels(clustering_all_diss[j]),        # The J-TH cluster
                    'clustering_best': stability_labels(clustering_all_diss[best]),  # The best-TH clustering
                    'aggWeights': ac['aggWeights']
                })
                tab = df.groupby(['clustering_j', 'clustering_best'])['aggWeights'].sum().unstack(fill_value=0)

                val = [adjustedRandIndex(tab), jaccardCoef(tab)]
                return val

            arilist = []

            if method in ["noise", "fuzzy"]:
                for j in range(R):
                    val = process_task(j, clustering_all_diss, ac, best)
                    arilist.append(val)
            else:
                arilist = Parallel(n_jobs=-1)(
                    delayed(process_task)(j, clustering_all_diss, ac, best) for j in range(R))

            arimatrix = np.vstack(arilist)
            arimatrix = pd.DataFrame(arimatrix, columns=["ARI", "JC"])
            ari08 = np.sum(arimatrix.iloc[:, 0] >= 0.8)
            jc08 = np.sum(arimatrix.iloc[:, 1] >= 0.8)

        else:
            arimatrix = np.nan
            ari08 = np.nan
            jc08 = np.nan

        _clustering = clustering_all_diss[best]

        if method in ("fuzzy", "noise"):
            disagclust = _clustering[np.asarray(ac["disaggIndex"], dtype=int) - 1, :]
        else:
            disagclust = np.full(seqdata.seqdata.shape[0], -1, dtype=float)
            for row_idx, agg_idx in enumerate(np.asarray(ac["disaggIndex"], dtype=int) - 1):
                disagclust[row_idx] = _clustering[agg_idx] + 1

        evol_diss = np.maximum.accumulate(objective) if _criteria in ["ams", "pbm"] else np.minimum.accumulate(objective)

        medoids_agg = np.asarray(med_all_diss[best], dtype=int)
        medoids_original = np.asarray(ac["aggIndex"], dtype=int)[medoids_agg] - 1

        # Store the best solution and evaluations of the others
        bestcluster = {
            "medoids": medoids_original,
            "medoids_agg": medoids_agg,
            "clustering": disagclust,
            "evol_diss": evol_diss,
            "iter_objective": objective,
            "objective": objective[best],
            "iteration": best,
            "arimatrix": arimatrix,
            "criteria": _criteria,
            "method": method,
            "avg_dist": mean_all_diss[best],
            "pbm": pbm_all[best],
            "db": db_all[best],
            "xb": xb_all[best],
            "ams": ams_all[best],
            "ari08": ari08,
            "jc08": jc08,
            "R": R,
            "k": k_value
        }

        if method == "representativeness":
            refseq = [list(range(len(agseqdata))), medoids_agg.tolist()]
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull):
                    states = np.arange(1, len(seqdata.states) + 1).tolist()
                    agseqdata_wrapped = SequenceData(agseqdata, time=seqdata.time, states=states)
                    dist_args["seqdata"] = agseqdata_wrapped
                    dist_args["refseq"] = refseq
                    diss2 = get_distance_matrix(opts=dist_args)
                    del dist_args["refseq"]
            diss2 = diss2.to_numpy(dtype=float)
            if np.any(~np.isfinite(diss2)) or np.any(diss2 < 0):
                raise ValueError("[!] Representative distances must be finite and non-negative.")
            observed_ref_max = float(np.max(diss2)) if diss2.size else 0.0
            if observed_ref_max > max_dist and not np.isclose(
                observed_ref_max,
                max_dist,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "[!] max.dist must be at least the maximum distance to selected representatives."
                )
            bestcluster["representativeness"] = np.clip(1.0 - diss2 / max_dist, 0.0, 1.0)
            disagclust = bestcluster["representativeness"][np.asarray(ac["disaggIndex"], dtype=int) - 1, :]
            bestcluster["clustering"] = disagclust

        # Store computed cluster quality
        kresult = {
            "k": k_value,
            "k_index": k_index,
            "criteria": criteria,
            "stats": [bestcluster["avg_dist"], bestcluster["pbm"], bestcluster["db"], bestcluster["xb"],
                      bestcluster["ams"], bestcluster["ari08"], bestcluster["jc08"], best],
            "bestcluster": bestcluster
        }

        kret.append(kresult)

    def claraObj(kretlines, method, kvals, kret, seqdata):
        matrix_valued = method in ("fuzzy", "noise", "representativeness")
        if matrix_valued:
            clustering = np.empty((seqdata.seqdata.shape[0], len(kvals)), dtype=object)
            clustering.fill(None)
        else:
            clustering = np.full((seqdata.seqdata.shape[0], len(kvals)), -1, dtype=float)
        clustering = pd.DataFrame(clustering)
        clustering.columns = [f"Cluster {val}" for val in kvals]
        clustering.index = seqdata.ids

        ret = {
            "kvals": kvals,
            "clara": {},
            "clustering": clustering,
            "stats": np.full((len(kvals), 8), -1, dtype=float)
        }

        for i in kretlines:
            k = kret[i].get('k_index', kret[i]['k'] - 2)
            ret['stats'][k, :] = np.array(kret[i]['stats'])
            ret['clara'][k] = kret[i]['bestcluster']

            best_clustering = kret[i]['bestcluster']['clustering']
            if matrix_valued:
                ret['clustering'].iloc[:, k] = list(np.asarray(best_clustering))
            else:
                ret['clustering'].iloc[:, k] = best_clustering

        ret['stats'] = pd.DataFrame(ret['stats'],
                                    columns=["Avg dist", "PBM", "DB", "XB", "AMS", "ARI>0.8", "JC>0.8", "Best iter"])
        ret['stats'].insert(0, "Number of Clusters", [f"Cluster {k}" for k in kvals])
        ret['stats']["k_num"] = kvals

        return ret

    if len(criteria) > 1:
        ret = {
            'param': {
                'criteria': criteria,
                'pam_combine': False,
                'all_criterias': criteria,
                'kvals': kvals,
                'method': method,
                'stability': stability
            }
        }

        for meth in criteria:
            indices = np.where(np.array([tup[1] for tup in kvalscriteria]) == meth)[0]
            ret[meth] = claraObj(kretlines=indices, method=method, kvals=kvals, kret=kret, seqdata=seqdata)

        allstats = {}

        for meth in criteria:
            stats = pd.DataFrame(ret[meth]['stats'])
            stats['criteria'] = meth

            allstats[meth] = stats

        ret['allstats'] = pd.concat(allstats.values(), ignore_index=False)
    else:

        ret = claraObj(kretlines=range(len(kvalscriteria)), method=method, kvals=kvals, kret=kret, seqdata=seqdata)

    print("  - Done.")

    return ret


def seqclara_range(
    seqdata,
    R=100,
    sample_size=None,
    kvals=None,
    seqdist_args=None,
    method="crisp",
    **kwargs,
):
    """WeightedCluster-compatible alias for :func:`clara`."""
    return clara(
        seqdata,
        R=R,
        sample_size=sample_size,
        kvals=kvals,
        dist_args=seqdist_args,
        method=method,
        **kwargs,
    )


if __name__ == '__main__':
    from sequenzo import *  # Social sequence analysis
    import pandas as pd  # Import necesarry packages

    # TODO : clara 返回的隶属矩阵要转置一下，因为plot_sequence_index里的参数id_group_df：cluster id 是行，id 是列

    # ===============================
    #             Sohee
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/data_and_output/orignal data/sohee/sequence_data.csv')
    # time_list = list(df.columns)[1:133]
    # states = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # # states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # labels = ['FT+WC', 'FT+BC', 'PT+WC', 'PT+BC', 'U', 'OLF']
    # sequence_data = SequenceData(df, time=time_list, time_type="age", states=states, labels=labels, id_col="PID")

    # om.to_csv("D:/college/research/QiQi/sequenzo/files/sequenzo_Sohee_string_OM_TRATE.csv", index=True)

    # ===============================
    #             kass
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/files/orignal data/kass/wide_civil_final_df.csv')
    # time_list = list(df.columns)[1:]
    # states = ['Extensive Warfare', 'Limited Violence', 'No Violence', 'Pervasive Warfare', 'Prolonged Warfare',
    #           'Serious Violence', 'Serious Warfare', 'Sporadic Violence', 'Technological Warfare', 'Total Warfare']
    # sequence_data = SequenceData(df, time=time_list, time_type="year", states=states, id_col="COUNTRY")

    # ===============================
    #             CO2
    # ===============================
    # df = pd.read_csv("D:/country_co2_emissions_missing.csv")
    # time = list(df.columns)[1:]
    # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    # sequence_data = SequenceData(df, time_type="age", time=time, id_col="country", states=states)

    # ===============================
    #            detailed
    # ===============================
    # df = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/sampled_data_sets/detailed_data/sampled_1000_data.csv")
    # time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']],
    #                              time=time, id_col="worker_id", states=states)

    # ===============================
    #             broad
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/broad_data/sampled_1000_data.csv")
    # time = list(df.columns)[4:]
    # states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5']],
    #                              time_type="age", time=time, id_col="worker_id", states=states)

    df = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/not_real_detailed_data/synthetic_detailed_U5_N1000.csv")
    _time = list(df.columns)[2:]
    states = ["Data", "Data science", "Hardware", "Research", "Software", "Support & test", "Systems & infrastructure"]
    df = df[['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]
    sequence_data = SequenceData(df, time=_time, id_col="id", states=states)

    result = clara(sequence_data,
                   R=250,
                   sample_size=500,
                   kvals=range(2, 6),
                   criteria=['distance'],
                   dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
                   stability=True)

    # print(result)
    print(result['stats'])
