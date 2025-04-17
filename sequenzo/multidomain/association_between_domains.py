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
    - Interpretation of association strength
    - Optimized performance using NumPy and SciPy
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import itertools
import sys


def _chi_cramers_v(xtab, chi2, df):
    """
    Computes Cramer's V and its associated p-value.

    Parameters:
        xtab (np.ndarray): Contingency table.
        chi2 (float): Chi-square statistic.
        df (int): Degrees of freedom.

    Returns:
        tuple: (Cramer's V value, p-value)
    """
    n = xtab.sum()
    nr, nc = xtab.shape
    min_dim = min(nr, nc)
    V = np.sqrt(chi2 / (n * (min_dim - 1)))
    p_val = 1 - stats.chi2.cdf(chi2, df)
    return V, p_val


def _log_likelihood_ratio_test(xtab, struct_zero=True):
    """
    Computes the likelihood ratio test statistic for independence.

    Parameters:
        xtab (np.ndarray): Contingency table.
        struct_zero (bool): Adjust degrees of freedom for structural zeros.

    Returns:
        tuple: (LRT statistic, degrees of freedom, p-value)
    """
    observed = xtab.copy()
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_totals @ col_totals / total

    if struct_zero:
        zero_mask = (observed == 0)
        expected[zero_mask] = 1
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


def _classify_strength(v):
    """
    Classifies the strength of association based on Cramer's V value.

    Parameters:
        v (float): Cramer's V statistic (0 to 1).

    Returns:
        str: Strength level as a descriptive label.
    """
    if v < 0.1:
        print(
            "\n[!] Note: A Cramer's V below 0.1 suggests no strong linear association. "
            "However, non-linear dependencies may still exist and are not captured by Cramer's V."
        )
        return "None"
    elif v < 0.3:
        return "Weak"
    elif v < 0.5:
        return "Moderate"
    else:
        return "Strong"


def _pvalue_to_stars(p):
    """
    Converts a p-value to significance stars.
    Parameters:
        p (float): P-value
    Returns:
        str: Significance stars string
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def _explain_association(result_df):
    """
    Generates human-readable explanations from result DataFrame.

    Parameters:
        result_df (pd.DataFrame): Result table with Cramer's V values.

    Returns:
        list of str: Explanation strings.
    """
    explanations = []
    for idx, row in result_df.iterrows():
        v = row.get("v", np.nan)
        if not pd.isna(v):
            strength = _classify_strength(v)
            text = f"{idx.replace('_with_', ' vs ')}: {strength} association (Cramer's V = {v:.3f})"
            explanations.append(text)
    return explanations


def _attach_explanations(result_df):
    """
    Adds interpretation column ('strength') to result DataFrame.

    Parameters:
        result_df (pd.DataFrame): Original result table.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns.
    """
    result_df["strength"] = result_df["v"].apply(
        lambda v: _classify_strength(v) if not pd.isna(v) else ""
    )
    return result_df


def _show_full_dataframe(df):
    """
    Print full DataFrame in full width (for terminal or notebook).
    Handles environments without IPython gracefully.

    Parameters:
        df (pd.DataFrame): Original result table.

    Returns:
        pd.DataFrame: Full DataFrame.
    """

    # print(df.to_string(index=True))

    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.colheader_justify', 'left'):
        if 'ipykernel' in sys.modules:
            try:
                from IPython.display import display
                display(df)
            except ImportError:
                print(df.to_string(index=True))
        else:
            print(df.to_string(index=True))


def get_association_between_domains(seqdata_dom, assoc=("LRT", "V"), rep_method="overall",
                                    wrange=None, p_value=True, struct_zero=True, cross_table=False,
                                    with_missing=False, weighted=True, dnames=None,
                                    explain=True):
    """
    Computes pairwise associations between multiple sequence domains using statistical tests.

    Parameters:
        seqdata_dom (list): List of SequenceData objects, one per domain.
        assoc (tuple): Which association measures to compute: "LRT", "V", or both.
        rep_method (str): Method to determine which sequences to compare (currently only "overall").
        wrange (tuple or None): Not implemented yet (for time window comparison).
        p_value (bool): Whether to compute p-values.
        struct_zero (bool): Whether to treat structural zeros as affecting degrees of freedom.
        cross_table (bool): If True, attach cross-tabulations to result attributes.
        with_missing (bool): Whether to include rows/cols that only contain missing or void.
        weighted (bool): Whether to apply sequence weights from the first domain.
        dnames (list or None): Names of the domains. If None, will auto-name them as Dom1, Dom2, ...
        explain (bool): If True, add interpretation columns and print explanations.

    Returns:
        pd.DataFrame: A result table (rows = domain pairs; columns = df, LRT, v, p-values, etc.),
                      possibly with `strength` and `explanation` columns when `explain=True`.
                      If `cross_table=True`, the cross tables are stored in the `.attrs` dictionary.
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
        tabname = f"{name1} vs {name2}"
        tabnames.append(tabname)

        xtab = d1.get_xtabs(d2, weighted=weighted)

        if not with_missing:
            xtab = xtab[(xtab.sum(axis=1) > 0), :]
            xtab = xtab[:, (xtab.sum(axis=0) > 0)]

        res = {"df": None, "LRT": None, "p(LRT)": None, "v": None, "p(v)": None}

        if "LRT" in assoc:
            lrt, df, plrt = _log_likelihood_ratio_test(xtab, struct_zero)
            res["LRT"] = lrt
            res["df"] = df
            if p_value:
                res["p(LRT)"] = plrt

        if "V" in assoc:
            if res["df"] is None:
                _, df, _ = _log_likelihood_ratio_test(xtab, struct_zero)
                res["df"] = df
            chi2 = stats.chi2_contingency(xtab, correction=False)[0]
            v, pv = _chi_cramers_v(xtab, chi2, res["df"])
            res["v"] = v
            if p_value:
                res["p(v)"] = pv

        results.append(res)
        if cross_table:
            cross_tables[tabname] = xtab

    colnames = ["df", "LRT", "p(LRT)", "v", "p(v)"]
    result_matrix = np.full((len(results), len(colnames)), np.nan)
    for idx, res in enumerate(results):
        for col_idx, col in enumerate(colnames):
            if res[col] is not None:
                result_matrix[idx, col_idx] = res[col]

    result_df = pd.DataFrame(result_matrix, columns=colnames, index=tabnames)

    # Safely attach cross tables without causing Pandas printing issues
    if cross_table:
        # Only store *serializable* data to avoid print issues
        result_df.attrs["cross.tables"] = {
            k: xtab.tolist() for k, xtab in cross_tables.items()
        }
    else:
        # Completely clear attrs to avoid ValueError when printing
        result_df.attrs.clear()

    # After computing result_df
    if explain:
        result_df = _attach_explanations(result_df)

        result_df["p(LRT)"] = result_df["p(LRT)"].apply(
            lambda p: f"{p:.3f} {_pvalue_to_stars(p)}".strip() if not pd.isna(p) else ""
        )

        # Convert p(v) to string with stars
        result_df["p(v)"] = result_df["p(v)"].apply(
            lambda p: f"{p:.3f} {_pvalue_to_stars(p)}".strip() if not pd.isna(p) else ""
        )

    print("\n📜 Full results table:")
    _show_full_dataframe(result_df)

    print("\n📘 Column explanations:")
    print("  - df       : Degrees of freedom for the test (typically 1 for binary state sequences).")
    print("  - LRT      : Likelihood Ratio Test statistic (higher = stronger dependence).")
    print("  - p(LRT)   : p-value for LRT + significance stars: * (p<.05), ** (p<.01), *** (p<.001)")
    print("  - v        : Cramer's V statistic (0 to 1, measures association strength).")
    print("  - p(v)     : p-value for Cramer's V (based on chi-squared test) + significance stars: * (p<.05), ** (p<.01), *** (p<.001)")
    print("  - strength : Qualitative label for association strength based on Cramer's V:")
    print("               0.00–0.09 → None, 0.10–0.29 → Weak, 0.30–0.49 → Moderate, ≥0.50 → Strong")

