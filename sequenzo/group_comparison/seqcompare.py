"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqcompare.py
@Time    : 2026-02-15 11:17
@Desc    : 
Likelihood Ratio Test and Bayesian Information Criterion for comparing sets of sequences.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
import warnings
import time

from ..dissimilarity_measures.get_distance_matrix import get_distance_matrix
from ..define_sequence_data import SequenceData


def compute_bayesian_information_criterion_test(
    seqdata: Union[pd.DataFrame, List[pd.DataFrame]],
    seqdata2: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    squared: str = "LRTonly",
    weighted: bool = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    return compare_groups_overall(
        seqdata=seqdata,
        seqdata2=seqdata2,
        group=group,
        set_var=set_var,
        s=s,
        seed=seed,
        stat="BIC",
        squared=squared,
        weighted=weighted,
        opt=opt,
        BFopt=BFopt,
        method=method,
        **kwargs
    )


def compute_likelihood_ratio_test(
    seqdata: Union[pd.DataFrame, List[pd.DataFrame]],
    seqdata2: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    squared: str = "LRTonly",
    weighted: bool = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    return compare_groups_overall(
        seqdata=seqdata,
        seqdata2=seqdata2,
        group=group,
        set_var=set_var,
        s=s,
        seed=seed,
        stat="LRT",
        squared=squared,
        weighted=weighted,
        opt=opt,
        BFopt=BFopt,
        method=method,
        **kwargs
    )


def compare_groups_overall(
    seqdata: Union[pd.DataFrame, SequenceData, List[pd.DataFrame], List[SequenceData]],
    seqdata2: Optional[Union[pd.DataFrame, SequenceData, List[pd.DataFrame], List[SequenceData]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    stat: str = "all",
    squared: Union[bool, str] = "LRTonly",
    weighted: Union[bool, str] = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    ptime_begin = time.time()

    if seqdata2 is None and group is None:
        raise ValueError("[!] 'seqdata2' and 'group' cannot both be None!")
    if set_var is not None and group is None:
        raise ValueError("[!] 'set_var' not None while 'group' is None!")

    if isinstance(weighted, str):
        if weighted != "by.group":
            raise ValueError("[!] weighted must be logical or 'by.group'")
        weight_by = weighted
        weighted = True
    else:
        weight_by = "global"

    if isinstance(squared, bool):
        LRTpow = 1
    else:
        if squared != "LRTonly":
            raise ValueError("[!] squared must be logical or 'LRTonly'")
        LRTpow = 2
        squared = False

    is1_seqdata = isinstance(seqdata, (pd.DataFrame, SequenceData))
    is2_seqdata = isinstance(seqdata2, (pd.DataFrame, SequenceData)) if seqdata2 is not None else False

    if isinstance(seqdata, list):
        if is2_seqdata or (seqdata2 is not None and not isinstance(seqdata2, list)) or len(seqdata) != len(seqdata2):
            raise ValueError("[!] When 'seqdata' is a list, seqdata2 must be a list of same length")
        for i, (sd1, sd2) in enumerate(zip(seqdata, seqdata2)):
            if not isinstance(sd1, (pd.DataFrame, SequenceData)) or not isinstance(sd2, (pd.DataFrame, SequenceData)):
                raise TypeError(f"[!] At least one element of the seqdata lists at index {i} is not a DataFrame or SequenceData!")
    elif not is1_seqdata:
        raise TypeError("[!] If not a list, 'seqdata' must be a DataFrame or SequenceData object")
    elif seqdata2 is not None and not is2_seqdata:
        raise TypeError("[!] If not a list, 'seqdata2' must be a DataFrame or SequenceData object")

    if isinstance(seqdata, list):
        seqdata_df = None
        seqdata_original = seqdata
    elif isinstance(seqdata, SequenceData):
        seqdata_df = seqdata.data[seqdata.time].copy()
        seqdata_original = seqdata
    else:
        seqdata_df = seqdata.copy()
        seqdata_original = seqdata

    if seqdata2 is not None:
        if isinstance(seqdata2, SequenceData):
            seqdata2_df = seqdata2.data[seqdata2.time].copy()
        elif isinstance(seqdata2, pd.DataFrame):
            seqdata2_df = seqdata2.copy()
        else:
            seqdata2_df = seqdata2
    else:
        seqdata2_df = None

    valid_stats = ["LRT", "BIC", "all"]
    if not all(s_ in valid_stats for s_ in ([stat] if isinstance(stat, str) else stat)):
        raise ValueError(f"[!] Bad stat value, must be one of {', '.join(valid_stats)}")
    if stat == "all":
        is_LRT = is_BIC = True
    else:
        is_LRT = "LRT" in ([stat] if isinstance(stat, str) else stat)
        is_BIC = "BIC" in ([stat] if isinstance(stat, str) else stat)

    if not is1_seqdata:
        seq1 = seqdata
        seq2 = seqdata2
    elif is1_seqdata and seqdata2 is not None:
        seq1 = [seqdata_df]
        seq2 = [seqdata2_df]
    else:
        gvar = np.asarray(group)
        if set_var is not None:
            setvar = np.asarray(set_var)
            inotna = np.where(~(pd.isna(gvar) | pd.isna(setvar)))[0]
            setvar = pd.Categorical(setvar[inotna])
            lev_set = setvar.categories.tolist()
        else:
            inotna = np.where(~pd.isna(gvar))[0]
        n_removed = len(gvar) - len(inotna)
        if n_removed > 0:
            print(f"[!!] {n_removed} sequences removed because of NA values of the grouping variable(s)")
        gvar = pd.Categorical(gvar[inotna])
        lev_g = gvar.categories.tolist()
        if len(lev_g) == 1:
            raise ValueError("[!] There is only one group among valid cases!")
        if len(lev_g) > 2:
            raise ValueError("[!] Currently seqcompare supports only 2 groups!")
        seqdata_filtered = seqdata_df.iloc[inotna, :]
        seq1, seq2 = [], []
        if set_var is None:
            seq1.append(seqdata_filtered[gvar == lev_g[0]])
            seq2.append(seqdata_filtered[gvar == lev_g[1]])
        else:
            for lev in lev_set:
                mask1 = (gvar == lev_g[0]) & (setvar == lev)
                mask2 = (gvar == lev_g[1]) & (setvar == lev)
                seq1.append(seqdata_filtered[mask1])
                seq2.append(seqdata_filtered[mask2])

    G = len(seq1)
    n = np.zeros((G, 2), dtype=int)
    seq_a, seq_b = [], []
    for i in range(G):
        n1, n2 = len(seq1[i]), len(seq2[i])
        if n1 >= n2:
            n[i, :] = [n1, n2]
            seq_a.append(seq1[i]); seq_b.append(seq2[i])
        else:
            n[i, :] = [n2, n1]
            seq_a.append(seq2[i]); seq_b.append(seq1[i])
    n_n = n.min(axis=1)

    if s > 0:
        m_n = n.max(axis=1)
        f_n1 = np.floor(s / m_n).astype(int)
        ff_n1 = np.maximum(1, f_n1)
        r_n1 = np.where(s < m_n, s - (m_n % s), s - f_n1 * m_n)
        k_n = np.floor((ff_n1 * m_n + r_n1) / n_n).astype(int)
        k_n[pd.isna(k_n)] = 0
        r_n2 = (ff_n1 * m_n + r_n1) - k_n * n_n
        r_n2[pd.isna(r_n2)] = 0

    nc = 4 if (is_LRT and is_BIC) else 2
    Results = np.full((G, nc), np.nan)
    multsple = False

    for i in range(G):
        if n_n[i] <= 0:
            continue
        if s == 0:
            r1 = np.arange(len(seq_a[i]))
            r2 = np.arange(len(seq_b[i])) + len(seq_a[i])
            combined_seqs_df = pd.concat([seq_a[i], seq_b[i]], ignore_index=True)
            weights_a = np.ones(len(seq_a[i]))
            weights_b = np.ones(len(seq_b[i]))
            if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, "weights") and weighted and group is not None:
                mask_a = gvar == lev_g[0]
                mask_b = gvar == lev_g[1]
                weights_a = seqdata_original.weights[inotna][mask_a]
                weights_b = seqdata_original.weights[inotna][mask_b]
            weights = np.concatenate([weights_a, weights_b])
            temp_states = seqdata_original.states if isinstance(seqdata_original, SequenceData) else sorted(combined_seqs_df.stack().dropna().unique().tolist())
            combined_seqs = SequenceData(combined_seqs_df, time=list(combined_seqs_df.columns), states=temp_states, weights=weights)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diss = get_distance_matrix(seqdata=combined_seqs, method=method, weighted=weighted, **kwargs)
            diss = diss.values if isinstance(diss, pd.DataFrame) else np.asarray(diss)
            Results[i, :] = _seqxcomp(r1, r2, diss, weights, is_LRT, is_BIC, squared, weighted, weight_by, LRTpow)
        else:
            np.random.seed(seed)
            mni = n.max(axis=1)[i]
            r_s1_flat = np.concatenate([np.random.permutation(np.repeat(np.arange(mni), ff_n1[i])), np.random.choice(mni, r_n1[i], replace=False)])
            r_s2_flat = np.concatenate([np.random.permutation(np.repeat(np.arange(n_n[i]), k_n[i])), np.random.choice(n_n[i], r_n2[i], replace=False)])
            num_samples_1 = len(r_s1_flat) // s
            num_samples_2 = len(r_s2_flat) // s
            r_s1 = r_s1_flat[:num_samples_1 * s].reshape(num_samples_1, s)
            r_s2 = r_s2_flat[:num_samples_2 * s].reshape(num_samples_2, s)
            opt_i = 1 if (opt is None and (len(seq_a[i]) + len(seq_b[i])) > 2 * s) else (2 if opt is None else opt)

            if opt_i == 2:
                combined_seqs_df = pd.concat([seq_a[i], seq_b[i]], ignore_index=True)
                weights_a = np.ones(len(seq_a[i]))
                weights_b = np.ones(len(seq_b[i]))
                if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, "weights") and weighted and group is not None:
                    mask_a = gvar == lev_g[0]
                    mask_b = gvar == lev_g[1]
                    weights_a = seqdata_original.weights[inotna][mask_a]
                    weights_b = seqdata_original.weights[inotna][mask_b]
                weights = np.concatenate([weights_a, weights_b])
                temp_states = seqdata_original.states if isinstance(seqdata_original, SequenceData) else sorted(combined_seqs_df.stack().dropna().unique().tolist())
                combined_seqs = SequenceData(combined_seqs_df, time=list(combined_seqs_df.columns), states=temp_states, weights=weights)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    diss_full = get_distance_matrix(seqdata=combined_seqs, method=method, weighted=weighted, **kwargs)
                diss_full = diss_full.values if isinstance(diss_full, pd.DataFrame) else np.asarray(diss_full)

            multsple = r_s1.shape[0] > 1 or multsple
            t = np.zeros((r_s1.shape[0], nc))
            for j in range(r_s1.shape[0]):
                if opt_i == 2:
                    r1 = r_s1[j, :]
                    r2 = r_s2[j, :] + len(seq_a[i])
                    diss = diss_full
                    weights_a = np.ones(len(r_s1[j, :]))
                    weights_b = np.ones(len(r_s2[j, :]))
                    if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, "weights") and weighted and group is not None:
                        mask_a = gvar == lev_g[0]
                        mask_b = gvar == lev_g[1]
                        weights_all_a = seqdata_original.weights[inotna][mask_a]
                        weights_all_b = seqdata_original.weights[inotna][mask_b]
                        weights_a = weights_all_a[np.asarray(r_s1[j, :]).flatten()]
                        weights_b = weights_all_b[np.asarray(r_s2[j, :]).flatten()]
                    weights = np.concatenate([weights_a, weights_b])
                else:
                    indices_a = np.asarray(r_s1[j, :]).flatten().tolist()
                    indices_b = np.asarray(r_s2[j, :]).flatten().tolist()
                    seqA_df = seq_a[i].iloc[indices_a, :]
                    seqB_df = seq_b[i].iloc[indices_b, :]
                    seqAB_df = pd.concat([seqA_df, seqB_df], ignore_index=True)
                    wA = np.ones(len(seqA_df)); wB = np.ones(len(seqB_df))
                    if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, "weights") and weighted and group is not None:
                        mask_a = gvar == lev_g[0]
                        mask_b = gvar == lev_g[1]
                        mask_a_indices = np.where(mask_a)[0]
                        mask_b_indices = np.where(mask_b)[0]
                        sampled_indices_a = mask_a_indices[np.asarray(r_s1[j, :]).flatten()]
                        sampled_indices_b = mask_b_indices[np.asarray(r_s2[j, :]).flatten()]
                        wA = seqdata_original.weights[inotna][sampled_indices_a]
                        wB = seqdata_original.weights[inotna][sampled_indices_b]
                    weights = np.concatenate([wA, wB])
                    temp_states = seqdata_original.states if isinstance(seqdata_original, SequenceData) else sorted(seqAB_df.stack().dropna().unique().tolist())
                    seqAB = SequenceData(seqAB_df, time=list(seqAB_df.columns), states=temp_states, weights=weights)
                    r1 = np.arange(len(r_s1[j, :]))
                    r2 = np.arange(len(r_s2[j, :])) + len(r_s1[j, :])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        diss = get_distance_matrix(seqdata=seqAB, method=method, weighted=weighted, **kwargs)
                    diss = diss.values if isinstance(diss, pd.DataFrame) else np.asarray(diss)
                t[j, :] = _seqxcomp(r1, r2, diss, weights, is_LRT, is_BIC, squared, weighted, weight_by, LRTpow)
            Results[i, :] = t.mean(axis=0)

    colnames = []
    if is_LRT:
        colnames.extend(["LRT", "p-value"])
    if is_BIC:
        if BFopt is None and multsple:
            BF2 = np.exp(Results[:, nc - 1] / 2)
            Results = np.column_stack([Results, BF2])
            colnames.extend(["Delta BIC", "Bayes Factor (Avg)", "Bayes Factor (From Avg BIC)"])
        elif BFopt == 1 and multsple:
            colnames.extend(["Delta BIC", "Bayes Factor (Avg)"])
        elif BFopt == 2 and multsple:
            BF2 = np.exp(Results[:, nc - 1] / 2)
            Results[:, nc] = BF2
            colnames.extend(["Delta BIC", "Bayes Factor (From Avg BIC)"])
        else:
            colnames.extend(["Delta BIC", "Bayes Factor"])

    Results = pd.DataFrame(Results, columns=colnames)
    if set_var is not None:
        Results.index = lev_set
    ptime_end = time.time()
    print(f"elapsed time: {ptime_end - ptime_begin:.3f} seconds")
    return Results.values


def _seqxcomp(
    r1: np.ndarray,
    r2: np.ndarray,
    diss: np.ndarray,
    weights: np.ndarray,
    is_LRT: bool,
    is_BIC: bool,
    squared: bool,
    weighted: bool,
    weight_by: str,
    LRTpow: int
) -> np.ndarray:
    n1, n2 = len(r1), len(r2)
    n0 = n1 + n2
    weighted = weighted and weights is not None
    if weighted:
        w1 = weights[r1]; w2 = weights[r2]
        if weight_by == "by.group":
            w1 = n1 / w1.sum() * w1
            w2 = n2 / w2.sum() * w2
        w = np.concatenate([w1, w2])
    else:
        w = np.ones(n0); w1 = np.ones(n1); w2 = np.ones(n2)

    diss_array = diss.values if isinstance(diss, pd.DataFrame) else np.asarray(diss)
    r_combined = np.concatenate([r1, r2])
    dist_S = _disscenter(diss_array[np.ix_(r_combined, r_combined)], weights=w, squared=squared)
    dist_S1 = _disscenter(diss_array[np.ix_(r1, r1)], weights=w1, squared=squared)
    dist_S2 = _disscenter(diss_array[np.ix_(r2, r2)], weights=w2, squared=squared)
    SS = (w * (dist_S ** LRTpow)).sum()
    SS1 = (w1 * (dist_S1 ** LRTpow)).sum()
    SS2 = (w2 * (dist_S2 ** LRTpow)).sum()
    LRT = n0 * (np.log(SS / n0) - np.log((SS1 + SS2) / n0))
    res = []
    if is_LRT:
        from scipy.stats import chi2
        res.extend([LRT, chi2.sf(LRT, df=1)])
    if is_BIC:
        BIC = LRT - np.log(n0)
        BF = np.exp(BIC / 2)
        res.extend([BIC, BF])
    return np.array(res)


def _disscenter(
    diss: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> np.ndarray:
    if squared:
        diss = diss ** 2
    n = diss.shape[0]
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=np.float64)
    dist_center = np.zeros(n)
    for i in range(n):
        dist_center[i] = (weights * diss[i, :]).sum()
    weighted_mean = (weights * dist_center).sum() / weights.sum()
    return dist_center - weighted_mean / 2
