"""
@Author  : Yuqi Liang, Xinyi Li
@File    : overall_cross_sectional_entropy.py
@Time    : 2025/9/15 21:52
@Desc    : States frequency by time unit

        The corresponding function name in TraMineR is seqstatd.R,
        with the source code available at: https://github.com/cran/TraMineR/blob/master/R/seqstatd.R
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sequenzo.define_sequence_data import SequenceData

def get_cross_sectional_entropy(
    seqdata: SequenceData,
    weighted: bool = True,
    norm: bool = True,
    return_format: str = "tidy",        # "tidy" | "wide" | "dict"
    include_effective_states: bool = True,
    add_topk: int = 1,                  # Mark top K dominant states at each time point
    round_decimals: int = 6
):
    """
    Cross-sectional state distribution by time with entropy and readable outputs.

    What you get in a tidy format:
        time   state   freq   entropy   per_time_entropy_norm   N_valid   rank   is_top
        1      A       0.645  0.380     0.380          2346.27   1      True
        ...

    Additional metrics:
        - per_time_entropy_norm: If norm=True, normalized by maximum entropy (|S|), range 0-1
        - effective_states (H_effective): exp(H), equivalent "effective number of states"
        - summary: Key interpretation points (entropy peaks/valleys, dominant state intervals, average entropy, etc.)

    Parameters maintain your semantics, with new return_format, add_topk etc. for better interpretability.

    Parameters
    ----------
    seqdata : SequenceData
        A sequence object created by the SequenceData function.
    weighted : bool, default True
        If True, the frequencies are weighted by the number of non-missing values at each time unit.
    norm : bool, default True
        If True, the entropy is normalized by maximum possible entropy.
    return_format : str, default "tidy"
        Return format: "tidy" for long-form table, "wide" for matrices, "dict" for original dict format.
    include_effective_states : bool, default True
        If True, calculate effective number of states (exp(entropy)).
    add_topk : int, default 1
        Mark top K dominant states at each time point.
    round_decimals : int, default 6
        Number of decimal places for rounding.

    Returns
    -------
    pd.DataFrame or dict
        Depending on return_format:
        - "tidy": Long-form DataFrame with interpretable columns
        - "wide": Dict with frequency matrix, entropy series, etc.
        - "dict": Original dict format (backward compatible)
    """

    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData.")

    # Basic metadata
    states_labels = list(seqdata.states)                # Human-readable state labels
    S = len(states_labels)
    T = seqdata.seqdata.shape[1]                        # Number of time points
    times = list(seqdata.seqdata.columns)

    # Color attributes
    cpal = seqdata.custom_colors

    # Weights
    # Also takes into account that in unweighted sequence objects created with
    # older TraMineR versions the weights attribute is a vector of 1
    # instead of NULL
    w = seqdata.weights if seqdata.weights is not None else np.ones(seqdata.seqdata.shape[0])
    if np.all(w == 1):
        weighted = False

    # Your data is usually encoded with 1..S; if internally already labels, we can map here
    # For compatibility: build a "value -> row index" lookup table
    # Try to support both numeric encoding (1..S) and labels themselves
    value_to_row = {v: i for i, v in enumerate(range(1, S+1))}
    label_to_row = {lab: i for i, lab in enumerate(states_labels)}

    # Frequency matrix (S x T)
    freq_counts = np.zeros((S, T), dtype=float)

    for j in range(T):
        col = seqdata.seqdata.iloc[:, j]
        for i in range(S):
            # Try both encoding and label matching
            mask_num = (col == (i+1))
            mask_lab = (col == states_labels[i])
            mask = mask_num | mask_lab
            if weighted:
                freq_counts[i, j] = w[mask].sum()
            else:
                freq_counts[i, j] = mask.sum()

    N_valid = freq_counts.sum(axis=0)                 # Valid weight/sample size per time point
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(freq_counts, N_valid, where=(N_valid>0))  # Frequencies

    # Entropy
    H = np.array([entropy(P[:, j][P[:, j] > 0]) if N_valid[j] > 0 else 0.0 for j in range(T)])

    if norm:
        Hmax = entropy(np.ones(S) / S) if S > 0 else 1.0
        H_norm = H / Hmax if Hmax > 0 else H
    else:
        H_norm = H

    # Effective number of states (highly interpretable: equivalent "how many equiprobable states")
    H_eff = np.exp(H) if include_effective_states else None

    # Organize output: wide format
    freq_df_wide = pd.DataFrame(P, index=states_labels, columns=times).round(round_decimals)
    entropy_s = pd.Series(H_norm if norm else H, index=times, name=("per_time_entropy_norm" if norm else "Entropy")).round(round_decimals)
    valid_s   = pd.Series(N_valid, index=times, name="N_valid").round(round_decimals)
    eff_s     = (pd.Series(H_eff, index=times, name="Effective States").round(round_decimals)
                 if include_effective_states else None)

    # Generate tidy table (interpretation-friendly)
    tidy = (
        freq_df_wide
        .reset_index()
        .melt(id_vars="index", var_name="time", value_name="freq")
        .rename(columns={"index": "state"})
        .sort_values(["time", "freq"], ascending=[True, False])
    )
    # Ranking + topK annotation
    tidy["rank"] = tidy.groupby("time")["freq"].rank(method="first", ascending=False).astype(int)
    if add_topk and add_topk > 0:
        tidy["is_top"] = tidy["rank"] <= add_topk
    else:
        tidy["is_top"] = False

    # Merge entropy/sample size/effective states
    tidy = tidy.merge(entropy_s.reset_index().rename(columns={"index": "time"}), on="time", how="left")
    tidy = tidy.merge(valid_s.reset_index().rename(columns={"index": "time"}), on="time", how="left")
    if eff_s is not None:
        tidy = tidy.merge(eff_s.reset_index().rename(columns={"index": "time"}), on="time", how="left")

    # Friendly column order
    cols = ["time", "state", "freq"]
    if norm:
        cols += ["per_time_entropy_norm"]
    else:
        cols += ["Entropy"]
    cols += ["N_valid"]
    if include_effective_states:
        cols += ["Effective States"]
    cols += ["rank", "is_top"]
    tidy = tidy[cols]

    # Summary: key statistics that can be explained in one sentence
    summary = {
        "states": states_labels,
        "n_states": S,
        "n_timepoints": T,
        "avg_entropy_norm": float(tidy["per_time_entropy_norm"].mean()) if norm else None,
        "avg_entropy": float((entropy_s if not norm else entropy_s * entropy(np.ones(S)/S)).mean()) if not norm else None,
        "peak_entropy_time": tidy.loc[tidy["per_time_entropy_norm" if norm else "Entropy"].idxmax(), "time"] if T > 0 else None,
        "lowest_entropy_time": tidy.loc[tidy["per_time_entropy_norm" if norm else "Entropy"].idxmin(), "time"] if T > 0 else None,
        "dominant_stability_ratio": float(tidy.query("rank==1")["freq"].mean()),  # Average proportion of dominant state
        "cpal": cpal
    }

    # Print descriptive statistics
    print("\n" + "="*70)
    print("Cross-Sectional Entropy Summary")
    print("="*70)
    print(f"[>] Number of states: {summary['n_states']}")
    print(f"[>] Number of time points: {summary['n_timepoints']}")
    print(f"[>] On average, the most common state accounts for {summary['dominant_stability_ratio']:.1%} of cases")
    print(f"[>] Entropy is highest at time point {summary['peak_entropy_time']}")
    print(f"[>] Entropy is lowest at time point {summary['lowest_entropy_time']}")
    if norm:
        print(f"[>] Average normalized entropy: {summary['avg_entropy_norm']:.3f} (range: 0 = fully concentrated, 1 = evenly distributed)")
    print("="*70 + "\n")

    # Compatible with different return formats
    if return_format == "tidy":
        tidy.attrs = {"summary": summary}
        return tidy
    elif return_format == "wide":
        out = {
            "Frequencies": freq_df_wide,
            "N_valid": valid_s,
            ("per_time_entropy_norm" if norm else "Entropy"): entropy_s
        }
        if eff_s is not None:
            out["Effective States"] = eff_s
        return out
    else:  # "dict" -- try to be more readable too
        res = {
            "Frequencies": freq_df_wide,
            "ValidStates": valid_s,
            "Entropy": entropy_s if not norm else None,
            "per_time_entropy_norm": entropy_s if norm else None,
            "Effective States": eff_s,
            "__attrs__": {
                "nbseq": float(valid_s.iloc[0]) if len(valid_s)>0 else None,
                "cpal": cpal,
                "xtlab": times,
                "xtstep": getattr(seqdata, "xtstep", None),
                "tick_last": getattr(seqdata, "tick_last", None),
                "weighted": weighted,
                "norm": norm,
                "summary": summary
            }
        }
        return res