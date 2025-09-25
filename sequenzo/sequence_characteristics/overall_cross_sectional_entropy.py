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
        raise ValueError(" [!] data is NOT a sequence object, see SequenceData function to create one.")

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

if __name__ == "__main__":
    # ===============================
    #             Sohee
    # ===============================
    df = pd.read_csv('D:/college/research/QiQi/sequenzo/data_and_output/orignal data/sohee/sequence_data.csv')
    time_list = list(df.columns)[1:133]
    states = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    labels = ['FT+WC', 'FT+BC', 'PT+WC', 'PT+BC', 'U', 'OLF']
    sequence_data = SequenceData(df, time=time_list, states=states, labels=labels, id_col="PID")
    res = get_cross_sectional_entropy(sequence_data)

    # ===============================
    #             kass
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/files/orignal data/kass/wide_civil_final_df.csv')
    # time_list = list(df.columns)[1:]
    # states = ['Extensive Warfare', 'Limited Violence', 'No Violence', 'Pervasive Warfare', 'Prolonged Warfare',
    #           'Serious Violence', 'Serious Warfare', 'Sporadic Violence', 'Technological Warfare', 'Total Warfare']
    # sequence_data = SequenceData(df, time=time_list, time_type="year", states=states, id_col="COUNTRY")
    # res = seqstatd(sequence_data)

    # ===============================
    #             CO2
    # ===============================
    # df = pd.read_csv("D:/country_co2_emissions_missing.csv")
    # _time = list(df.columns)[1:]
    # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    # sequence_data = SequenceData(df, time_type="age", time=_time, id_col="country", states=states)
    # res = seqstatd(sequence_data)

    # ===============================
    #            detailed
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/detailed_data/sampled_1000_data.csv")
    # _time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']],
    #                              time_type="age", time=_time, id_col="worker_id", states=states)
    # res = seqstatd(sequence_data)

    # ===============================
    #             broad
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/broad_data/sampled_1000_data.csv")
    # _time = list(df.columns)[4:]
    # states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5']],
    #                              time_type="age", time=_time, id_col="worker_id", states=states)
    # res = seqstatd(sequence_data)

    print(res['Frequencies'])
    print(res['ValidStates'])
    print(res['Entropy'])