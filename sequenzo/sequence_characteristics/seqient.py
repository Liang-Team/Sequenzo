"""
@Author  : 李欣怡
@File    : seqient.py
@Time    : 2025/9/23 19:44
@Desc    : Within Sequence Entropy
"""
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.stats import entropy

from sequenzo.define_sequence_data import SequenceData
from seqistatd import seqistatd

def seqient(seqdata, norm=True, base=np.e, silent=True):
    """
    TraMineR 默认不计算缺失值，但我们默认处理缺失值。
    如果想要和 TraMineR 的数据一致：
    （1）有缺失值时，TraMIneR
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError(" [!] data is NOT a sequence object, see SequenceData function to create one.")

    states = seqdata.states.copy()

    if not silent:
        print(f"  - computing within sequence entropy for {seqdata.seqdata.shape[0]} sequences and {len(states)} states ...")

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            iseqtab = seqistatd(seqdata=seqdata)

    ient = iseqtab.apply(lambda row: entropy(row, base=base), axis=1)

    if norm:
        maxent = np.log(len(states))
        ient = ient / maxent

    ient.columns = ['Entropy']
    ient.index = seqdata.seqdata.index

    return ient


if __name__ == "__main__":
    # ===============================
    #             Sohee
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/data_and_output/orignal data/sohee/sequence_data.csv')
    # time_list = list(df.columns)[1:133]
    # states = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # # states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # labels = ['FT+WC', 'FT+BC', 'PT+WC', 'PT+BC', 'U', 'OLF']
    # sequence_data = SequenceData(df, time=time_list, states=states, labels=labels, id_col="PID")
    # res = seqient(sequence_data)

    # ===============================
    #             kass
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/files/orignal data/kass/wide_civil_final_df.csv')
    # time_list = list(df.columns)[1:]
    # states = ['Extensive Warfare', 'Limited Violence', 'No Violence', 'Pervasive Warfare', 'Prolonged Warfare',
    #           'Serious Violence', 'Serious Warfare', 'Sporadic Violence', 'Technological Warfare', 'Total Warfare']
    # sequence_data = SequenceData(df, time=time_list, states=states, id_col="COUNTRY")
    # res = seqient(sequence_data)

    # ===============================
    #             CO2
    # ===============================
    # df = pd.read_csv("D:/country_co2_emissions_missing.csv")
    # _time = list(df.columns)[1:]
    # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    # sequence_data = SequenceData(df, time=_time, id_col="country", states=states)
    # res = seqient(sequence_data)

    # ===============================
    #            detailed
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/detailed_data/sampled_1000_data.csv")
    # _time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']],
    #                              time=_time, id_col="worker_id", states=states)
    # res = seqient(sequence_data)

    # ===============================
    #             broad
    # ===============================
    df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/broad_data/sampled_1000_data.csv")
    _time = list(df.columns)[4:]
    states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5']],
                                 time=_time, id_col="worker_id", states=states)
    res = seqient(sequence_data)

    print(res)
