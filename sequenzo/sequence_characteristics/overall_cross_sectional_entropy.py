"""
@Author  : 李欣怡
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

def get_cross_sectional_entropy(seqdata: SequenceData,
                                weighted: bool = True,
                                norm: bool = True):
    """
    Compute the states frequency by time unit.

    Parameters
    ----------
    seqdata : SequenceData
        A sequence object created by the SequenceData function.
    weighted : bool, default True
        If True, the frequencies are weighted by the number of non-missing values at each time unit.
    with_missing : bool, default False
        If True, missing values are included in the frequency computation.
    norm : bool, default True
        If True, the frequencies are normalized to sum to 1 at each time unit.

    Returns
    -------
    pd.DataFrame
        A dict, containing the frequency of each state at each time unit, validStates and Entropy.
    """

    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData function to create one.")

    # Retrieve the states, shape and colors
    states = seqdata.states.copy()
    statl = range(1, len(states) + 1)
    number_states = len(statl)

    number_seq = seqdata.seqdata.shape[1]

    cpal = seqdata.custom_colors

    # SequenceData already added missing values into states and colors internally
    # if seqdata.ismissing:
    #     statl.append(seqdata.missing_value)
    #     col.append(seqdata.missing_color)

    sd = pd.DataFrame(np.zeros((number_states, number_seq)), index=states, columns=seqdata.seqdata.columns)

    # Weights
    weights = seqdata.weights if seqdata.weights is not None else np.ones(seqdata.seqdata.shape[0])

    # Also takes into account that in unweighted sequence objects created with
    # older TraMineR versions the weights attribute is a vector of 1
    # instead of NULL
    if np.all(weights == 1):
        weighted = False

    for i in range(number_states):
        for j in range(number_seq):
            sd.iloc[i, j] = np.sum(weights[(seqdata.seqdata.iloc[:, j] == statl[i]).values])

    N = sd.sum(axis=0)
    sd = sd.div(N, axis=1)

    E = sd.apply(lambda col: entropy(col[col > 0]), axis=0)

    # Maximum entropy is the entropy of the alphabet
    if norm:
        E_max = entropy(np.ones(number_states) / number_states)
        E = E / E_max

    res = {
        "Frequencies": sd,
        "ValidStates": N,
        "Entropy": E
    }

    res_attrs = {
        "nbseq": np.sum(weights),
        "cpal": cpal,
        "xtlab": list(seqdata.seqdata.columns),
        "xtstep": getattr(seqdata, "xtstep", None),
        "tick_last": getattr(seqdata, "tick_last", None),
        "weighted": weighted,
        "norm": norm
    }

    res["__attrs__"] = res_attrs

    return res