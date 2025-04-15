"""
@Author  : Yuqi Liang 梁彧祺
@File    : association_between_domains.py
@Time    : 14/04/2025 21:15
@Desc    : 
    This module provides functionality for measuring the association
    between multiple domains of sequence data. It is a Python implementation
    of the R function `seqdomassoc`, and supports calculating statistical
    measures such as Cramer's V and the likelihood ratio test (LRT)
    between pairs of sequence dimensions.

    Currently, only the "overall" comparison method is supported,
    which compares sequences position by position. Support for
    representative sequences and group medoids can be added in the future.

    Key features:
    - Pairwise association analysis between sequence domains
    - Support for weighted sequences
    - Cramer's V and LRT calculation with p-values
    - Cross-tabulation matrix extraction
    - Optimized performance using NumPy and SciPy
"""
import numpy as np
import scipy.stats as stats
import itertools
import pandas as pd


def chi_cramers_v(xtab, chi2, df):
    n = xtab.sum()
    nr, nc = xtab.shape
    min_dim = min(nr, nc)
    V = np.sqrt(chi2 / (n * (min_dim - 1)))
    p_val = 1 - stats.chi2.cdf(chi2, df)
    return V, p_val


def log_likelihood_ratio_test(xtab, struct_zero=True):
    observed = xtab.copy()
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    expected = row_totals @ col_totals / total

    if struct_zero:
        zero_mask = (observed == 0)
        expected[zero_mask] = 1  # Prevent log(0)
        observed[zero_mask] = 1

    with np.errstate(divide='ignore', invalid='ignore'):
        lrt_terms = np.where(observed > 0, observed * np.log(observed / expected), 0)
    lrt_stat = 2 * np.sum(lrt_terms)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    if struct_zero:
        df -= np.sum(observed == 0)
        df = max(df, 1)
    p_val = 1 - stats.chi2.cdf(lrt_stat, df)
    return lrt_stat, df, p_val


def get_association_between_domains(seqdata_dom, assoc=("LRT", "V"), rep_method="overall",
                                    wrange=None, p_value=True, struct_zero=True, cross_table=False,
                                    with_missing=False, weighted=True, dnames=None):
    """
    Python version of R's seqdomassoc (currently supports rep_method="overall").
    Parameters:
        seqdata_dom: list of SequenceData objects
        assoc: ("LRT", "V") or subset
        rep_method: only "overall" currently supported
        ...
    Returns:
        DataFrame with association measures between domain pairs
    """
    assoc = [a.upper() for a in assoc]
    valid_assoc = {"LRT", "V"}
    if not set(assoc).issubset(valid_assoc):
        raise ValueError(f"assoc must be a subset of {valid_assoc}")

    if len(seqdata_dom) < 2:
        raise ValueError("seqdata_dom must be a list of at least two SequenceData objects")

    if rep_method != "overall":
        raise NotImplementedError("Only rep_method='overall' is supported in this version")

    ndom = len(seqdata_dom)
    if dnames is None:
        dnames = [f"Dom{i + 1}" for i in range(ndom)]

    cross_tables = {}
    results = []
    tabnames = []

    for i, j in itertools.combinations(range(ndom), 2):
        d1, d2 = seqdata_dom[i], seqdata_dom[j]
        name1, name2 = dnames[i], dnames[j]
        tabname = f"{name1}_with_{name2}"
        tabnames.append(tabname)

        xtab = d1.get_xtabs(d2, weighted=weighted)

        # Remove rows/cols with all zeros (e.g., due to missing or voids)
        if not with_missing:
            xtab = xtab[(xtab.sum(axis=1) > 0), :]
            xtab = xtab[:, (xtab.sum(axis=0) > 0)]

        res = {"df": None, "LRT": None, "p(LRT)": None, "v": None, "p(v)": None}

        if "LRT" in assoc:
            lrt, df, plrt = log_likelihood_ratio_test(xtab, struct_zero)
            res["LRT"] = lrt
            res["df"] = df
            if p_value:
                res["p(LRT)"] = plrt

        if "V" in assoc:
            if res["df"] is None:
                _, df, _ = log_likelihood_ratio_test(xtab, struct_zero)
                res["df"] = df
            chi2 = stats.chi2_contingency(xtab, correction=False)[0]
            v, pv = chi_cramers_v(xtab, chi2, res["df"])
            res["v"] = v
            if p_value:
                res["p(v)"] = pv

        results.append(res)

        if cross_table:
            cross_tables[tabname] = xtab

    # Format result matrix
    colnames = ["df", "LRT", "p(LRT)", "v", "p(v)"]
    result_matrix = np.full((len(results), len(colnames)), np.nan)

    for idx, res in enumerate(results):
        for col_idx, col in enumerate(colnames):
            if res[col] is not None:
                result_matrix[idx, col_idx] = res[col]

    result_df = pd.DataFrame(result_matrix, columns=colnames, index=tabnames)

    if cross_table:
        result_df.attrs["cross.tables"] = cross_tables

    return result_df


if __name__ == '__main__':
    # 假设已经有三个 SequenceData 对象：
    # seq_child, seq_marr, seq_left

    # TODO: download biofam
    # TODO 先搞这个相关性，然后把其他三个衡量方法做好，然后再做可视化

    seq_child, seq_marr, seq_left = None, None, None

    result = get_association_between_domains(
        [seq_child, seq_marr, seq_left],
        assoc=["V", "LRT"],
        rep_method="overall",
        cross_table=True,
        weighted=True,
        dnames=["child", "marr", "left"]
    )

    print(result)
