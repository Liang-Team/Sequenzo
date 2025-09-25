"""
@Author  : 李欣怡
@File    : seqST.py
@Time    : 2025/9/24 14:09
@Desc    : Computes the sequence turbulence measure

        The corresponding function name in TraMineR is seqST.R,
        with the source code available at: https://github.com/cran/TraMineR/blob/master/R/seqST.R

"""
import os
from contextlib import redirect_stdout
import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength
from .simple_characteristics import seqsubsn
from .variance_of_spell_durations import get_spell_duration_variance

def turb(x):
    phi = x[0]
    s2_tx = x[1]
    s2max = x[2]

    Tux = np.log2(phi * ((s2max + 1) / (s2_tx + 1)))
    return Tux

def get_turbulence(seqdata, norm=False, silent=True, type=1):
    """
    Computes the sequence turbulence measure

    Parameters
    ----------
    seqdata : SequenceData
        A sequence object created by the SequenceData function.
    norm : bool, default True
        If True, the frequencies are normalized to sum to 1 at each time unit.
    silent : bool, default True
        If True, suppresses the output messages.
    type : int, default 1
        Type of spell duration variance to be used. Can be either 1 or 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one column containing the turbulence measure for each sequence.
    """

    if not hasattr(seqdata, 'seqdata'):
        raise ValueError(" [!] data is NOT a sequence object, see SequenceData function to create one.")

    if not silent:
        print(f"  - extracting symbols and durations ...")
    spells = seqdss(seqdata)

    if not silent:
        print(f"  - computing turbulence type {type} for {seqdata.seqdata.shape[0]} sequence(s) ...")
    phi = seqsubsn(spells, DSS=False, with_missing=True)

    if any(np.isnan(phi)):
        # 使用有限的大数值，避免转换警告
        # np.finfo(float).max 在NumPy 1.24+会触发"invalid value encountered in cast"警告
        large_but_finite = 1e15  # 足够大但不会导致溢出警告
        phi = np.where(np.isnan(phi), large_but_finite, phi)
        print("[!] There are missing values in the data. A very large number was used instead so the calculation can continue.")

    s2_tx = get_spell_duration_variance(seqdata=seqdata, type=type)
    s2_tx_max = s2_tx['vmax']
    s2_tx = s2_tx['result']

    tmp = pd.DataFrame({'phi': phi.flatten(), 's2_tx': s2_tx, 's2max': s2_tx_max})
    Tx = tmp.apply(lambda row: turb([row['phi'], row['s2_tx'], row['s2max']]), axis=1).to_numpy()

    if norm:
        alph = seqdata.states.copy()
        maxlength = max(seqlength(seqdata))
        nrep = -(-maxlength // len(alph))  # Ceiling division

        turb_seq = pd.DataFrame(np.array((alph * nrep)[:maxlength]).reshape(1, -1))
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                # 为 states 创建对应的 labels，需要特别处理 np.nan 的情况
                turb_labels = []
                for i, state in enumerate(alph):
                    if pd.isna(state):
                        turb_labels.append("Missing")
                    else:
                        turb_labels.append(f"State_{i}")
                turb_seq = SequenceData(turb_seq, time=list(range(turb_seq.shape[1])), states=alph, labels=turb_labels)

        if len(alph) > 1:
            turb_phi = seqsubsn(turb_seq, DSS=False, with_missing=True)
        else:
            turb_phi = 2

        if turb_phi.isna().any().any():
            turb_phi = 1e15  # 使用有限大数值避免转换警告
            print("[!] phi set as max float due to exceeding value when computing max turbulence.")

        turb_s2 = get_spell_duration_variance(turb_seq, type=type)
        turb_s2_max = turb_s2['vmax']
        turb_s2 = turb_s2['result']

        tmp = pd.DataFrame({'phi': turb_phi.iloc[:, 0], 's2_tx': turb_s2, 's2max': turb_s2_max})
        maxT = tmp.apply(lambda row: turb([row['phi'], row['s2_tx'], row['s2max']]), axis=1).to_numpy()

        Tx_zero = np.where(Tx < 1)[0]
        Tx = (Tx - 1) / (maxT - 1)
        if len(Tx_zero) > 0:
            Tx[Tx_zero, :] = 0

    Tx_df = pd.DataFrame(Tx, index=seqdata.seqdata.index, columns=['Turbulence'])
    return Tx_df

if __name__ == "__main__":

    from sequenzo import *
    
    # ===============================
    #             Sohee
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/data_and_output/orignal data/sohee/sequence_data.csv')
    # # df = pd.read_csv('/Users/lei/Documents/Sequenzo_all_folders/sequence_data_sources/sohee/sequence_data.csv')
    # time_list = list(df.columns)[1:133]
    # states = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # # states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # labels = ['FT+WC', 'FT+BC', 'PT+WC', 'PT+BC', 'U', 'OLF']
    # sequence_data = SequenceData(df, time=time_list, states=states, labels=labels, id_col="PID")
    # res = get_turbulence(sequence_data)

    # ===============================
    #             kass
    # ===============================
    # df = pd.read_csv('D:/college/research/QiQi/sequenzo/files/orignal data/kass/wide_civil_final_df.csv')
    # time_list = list(df.columns)[1:]
    # states = ['Extensive Warfare', 'Limited Violence', 'No Violence', 'Pervasive Warfare', 'Prolonged Warfare',
    #           'Serious Violence', 'Serious Warfare', 'Sporadic Violence', 'Technological Warfare', 'Total Warfare']
    # sequence_data = SequenceData(df, time=time_list, states=states, id_col="COUNTRY")
    # res = seqST(sequence_data)

    # ===============================
    #             CO2
    # ===============================
    # df = pd.read_csv("D:/country_co2_emissions_missing.csv")
    df = load_dataset('country_co2_emissions_local_deciles')
    df.to_csv("D:/country_co2_emissions_local_deciles.csv", index=False)
    _time = list(df.columns)[1:]
    # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    states = ['D1 (Very Low)', 'D10 (Very High)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
    sequence_data = SequenceData(df, time=_time, id_col="country", states=states)
    res = get_turbulence(sequence_data, norm=True, type=2)

    # ===============================
    #            detailed
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/detailed_data/sampled_1000_data.csv")
    # _time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']],
    #                              time=_time, id_col="worker_id", states=states)
    # res = seqST(sequence_data, norm=False, type=2)

    # ===============================
    #             broad
    # ===============================
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/sampled_data_sets/broad_data/sampled_1000_data.csv")
    # _time = list(df.columns)[4:]
    # states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5']],
    #                              time=_time, id_col="worker_id", states=states)
    # res = seqST(sequence_data, norm=True, type=2)

    print(res)