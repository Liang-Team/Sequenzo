"""
@Author  : 李欣怡
@File    : clara.py
@Time    : 2024/12/27 12:04
@Desc    : 
"""

import gc
import os
from contextlib import redirect_stdout
import warnings

from joblib import Parallel, delayed
import fastcluster
from scipy.special import comb
from itertools import product

from sequenzo.big_data.clara.utils.aggregatecases import *
from sequenzo.big_data.clara.utils.davies_bouldin import *
from sequenzo.big_data.clara.utils.k_medoids_once import *
from sequenzo.big_data.clara.utils.get_weighted_diss import *

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


def clara(seqdata, R=100, kvals=None, sample_size=None, method="crisp", dist_args=None,
          criteria=["distance"], stability=False, parallel=False, max_dist=None):

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

    allmethods = ["crisp"]
    if method.lower() not in [m.lower() for m in allmethods]:
        raise ValueError(f"[!] Unknown method {method}. Please specify one of the following: {', '.join(allmethods)}")

    if method.lower() == "representativeness" and max_dist is None:
        raise ValueError("[!] You need to set max.dist when using representativeness method.")

    allcriteria = ["distance", "db", "xb", "pbm", "ams"]
    if not all(c.lower() in [crit.lower() for crit in allcriteria] for c in criteria):
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
    agseqdata = seqdata.seqdata.iloc[ac['aggIndex'], :]
    # agseqdata.attrs['weights'] = None
    ac['probs'] = ac['aggWeights'] / number_seq
    print(f"  - OK ({len(ac['aggWeights'])} unique cases).")

    # Memory cleanup before parallel computation
    gc.collect()
    print("[>] Starting iterations...\n")

    def calc_pam_iter(circle, agseqdata, sample_size, kvals, ac):
        mysample = np.random.choice(len(agseqdata), size=sample_size, p=ac['probs'], replace=True)
        mysample = pd.DataFrame({'id': mysample})

        # Re-aggregate!
        ac2 = DataFrameAggregator().aggregate(mysample)
        data_subset = agseqdata.iloc[mysample.iloc[ac2['aggIndex'], 0], :]

        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                states = np.arange(1, len(seqdata.states) + 1).tolist()
                data_subset = SequenceData(data_subset,
                                           time_type=seqdata.time_type,
                                           time=seqdata.time,
                                           states=states)
                dist_args['seqdata'] = data_subset
                diss = get_distance_matrix(opts=dist_args)

        diss = diss.values
        _diss = diss.copy()
        _diss = get_weighted_diss(_diss, ac2['aggWeights'])
        hc = fastcluster.linkage(_diss, method='ward')
        del _diss

        # For each number of clusters
        allclust = []

        for k in kvals:
            # Weighted PAM clustering on subsample
            clustering = k_medoids_once(diss=diss, k=k, cluster_only=True, initialclust=hc, weights=ac2['aggWeights'])
            medoids = mysample.iloc[ac2['aggIndex'][np.unique(clustering)], :]
            medoids = medoids.to_numpy().flatten()

            del clustering

            # =====================================================
            # Compute Distances Between All Sequence to the Medoids
            # =====================================================
            refseq = [list(range(0, len(agseqdata))), medoids.tolist()]

            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    states = np.arange(1, len(seqdata.states) + 1).tolist()
                    agseqdata = SequenceData(agseqdata,
                                             time_type=seqdata.time_type,
                                             time=seqdata.time,
                                             states=states)
                    dist_args['seqdata'] = agseqdata
                    dist_args['refseq'] = refseq
                    diss2 = get_distance_matrix(opts=dist_args)

                    agseqdata = agseqdata.seqdata   # Restore scene

            # Compute two minimal distances are used for silhouette width
            # and other criterions
            diss2 = diss2.to_numpy()
            alphabeta = np.array([np.sort(row)[:2] for row in diss2])
            sil = (alphabeta[:, 1] - alphabeta[:, 0]) / np.maximum(alphabeta[:, 1], alphabeta[:, 0])


            # Allocate to clusters
            memb = np.argmin(diss2, axis=1)     # Each data point is assigned to its nearest cluster

            mean_diss = np.sum(alphabeta[:, 0] * ac['probs'])

            warnings.filterwarnings('ignore', category=RuntimeWarning)  # The ÷0 case is ignored
            db = davies_bouldin_internal(diss=diss2, clustering=memb, medoids=medoids, weights=ac['aggWeights'])['db']
            warnings.resetwarnings()

            pbm = ((1 / len(medoids)) * (np.max(diss2[medoids]) / mean_diss)) ** 2
            ams = np.sum(sil * ac['probs'])

            distmed = diss2[medoids, :]
            distmed_flat = distmed[np.triu_indices_from(distmed, k=1)]  # Take the upper triangular part
            minsep = np.min(distmed_flat)

            xb = mean_diss / minsep

            del alphabeta
            del sil
            del diss2
            del distmed
            del minsep

            allclust.append({
                'mean_diss': mean_diss,
                'db': db,
                'pbm': pbm,
                'ams': ams,
                'xb': xb,
                'clustering': memb,
                'medoids': medoids
            })

        del diss
        gc.collect()

        return allclust

    # Compute in parallel using joblib
    # output example :
    #         results[0] = all iter1's = [{k=2's}, {k=3's}, ... , {k=10's}]
    #         results[1] = all iter2's = [{k=2's}, {k=3's}, ... , {k=10's}]
    results = Parallel(n_jobs=-1)(
        delayed(calc_pam_iter)(circle=i, agseqdata=agseqdata, sample_size=sample_size, kvals=kvals, ac=ac) for i in range(R))

    print("[>] Aggregating iterations for each k values...")

    # output example :
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
        k = item[0]
        _criteria = item[1]

        mean_all_diss = [d['mean_diss'] for d in collected_data[k]]
        db_all = [d['db'] for d in collected_data[k]]
        pbm_all = [d['pbm'] for d in collected_data[k]]
        ams_all = [d['ams'] for d in collected_data[k]]
        xb_all = [d['xb'] for d in collected_data[k]]
        clustering_all_diss = [d['clustering'] for d in collected_data[k]]
        med_all_diss = [d['medoids'] for d in collected_data[k]]

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
            def process_task(j, clustering_all_diss, ac, best):
                df = pd.DataFrame({
                    'clustering_j': clustering_all_diss[j],        # The J-TH cluster
                    'clustering_best': clustering_all_diss[best],  # The best-TH clustering
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

        disagclust = np.full(seqdata.seqdata.shape[0], -1)
        for i, index in enumerate(ac["disaggIndex"]):
            disagclust[i] = _clustering[index] + 1      # 1-based index for clusters

        evol_diss = np.maximum.accumulate(objective) if _criteria in ["ams", "pbm"] else np.minimum.accumulate(objective)

        # Store the best solution and evaluations of the others
        bestcluster = {
            "medoids": ac["aggIndex"][med_all_diss[best]],
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
            "k": k
        }

        # Store computed cluster quality
        kresult = {
            "k": k+2,
            "criteria": criteria,
            "stats": [bestcluster["avg_dist"], bestcluster["pbm"], bestcluster["db"], bestcluster["xb"],
                      bestcluster["ams"], bestcluster["ari08"], bestcluster["jc08"], best],
            "bestcluster": bestcluster
        }

        kret.append(kresult)

    def claraObj(kretlines, method, kvals, kret, seqdata):
        clustering = np.full((seqdata.seqdata.shape[0], len(kvals)), -1)
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
            k = kret[i]['k'] - 2    # start from 0, not 2
            ret['stats'][k, :] = np.array(kret[i]['stats'])
            ret['clara'][k] = kret[i]['bestcluster']

            ret['clustering'].iloc[:, k] = kret[i]['bestcluster']['clustering']

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

    return ret


if __name__ == '__main__':
    from sequenzo import *  # Social sequence analysis
    import pandas as pd  # Import necesarry packages

    df = pd.read_csv('D:/country_co2_emissions_missing.csv')
    # df = pd.read_csv('/home/xinyi/data/detailed_data/sampled_1000_data.csv')

    time = list(df.columns)[1:]
    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    # time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # df = df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']]

    sequence_data = SequenceData(df, time=time, time_type="year", id_col="country", states=states)
    # sequence_data = SequenceData(df, time=time, time_type="age", id_col="worker_id", states=states)

    result = clara(sequence_data,
                   R=2,
                   sample_size=3000,
                   kvals=range(2, 6),
                   criteria=['distance'],
                   dist_args={"method": "OMspell", "sm": "CONSTANT", "indel": 1, "expcost": 1},
                   parallel=True,
                   stability=True)

    print(result['clustering'])

