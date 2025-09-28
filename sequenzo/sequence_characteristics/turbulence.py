"""
@Author  : Xinyi Li, Yuqi Liang
@File    : turbulence.py
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

def get_turbulence(seqdata, norm=False, silent=True, type=1, id_as_column=True):
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
    id_as_column : bool, default True
        If True, the ID will be included as a separate column instead of as the index.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one column containing the turbulence measure for each sequence.
        If id_as_column=True, also includes an ID column.
    """

    if not hasattr(seqdata, 'seqdata'):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData function to create one.")

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
        print("[!] One or more missing values were found after calculating the number of distinct subsequences. They have been replaced with a large number of 1e15 to ensure the calculation continues.")

    s2_tx = get_spell_duration_variance(seqdata=seqdata, type=type)
    s2_tx_max = s2_tx['vmax']
    s2_tx = s2_tx['result']

    # Extract phi values and ensure 1D array
    if hasattr(phi, 'iloc'):
        phi_values = phi.iloc[:, 0].values
    elif hasattr(phi, 'values'):
        phi_values = phi.values
    else:
        phi_values = phi
    
    # Ensure phi_values is 1D
    phi_values = np.asarray(phi_values).flatten()
    
    # Extract 1D arrays from s2_tx and s2_tx_max DataFrames
    s2_tx_values = s2_tx.iloc[:, 1].values if hasattr(s2_tx, 'iloc') else np.asarray(s2_tx).flatten()
    s2_tx_max_values = s2_tx_max.iloc[:, 1].values if hasattr(s2_tx_max, 'iloc') else np.asarray(s2_tx_max).flatten()
    
    tmp = pd.DataFrame({'phi': phi_values, 's2_tx': s2_tx_values, 's2max': s2_tx_max_values})
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

        if hasattr(turb_phi, 'isna') and turb_phi.isna().any().any():
            turb_phi = 1e15  # 使用有限大数值避免转换警告
            print("[!] phi set as max float due to exceeding value when computing max turbulence.")

        turb_s2 = get_spell_duration_variance(turb_seq, type=type)
        turb_s2_max = turb_s2['vmax']
        turb_s2 = turb_s2['result']

        # Extract turb_phi values and ensure 1D
        if hasattr(turb_phi, 'iloc'):
            phi_value = turb_phi.iloc[:, 0].values
        else:
            phi_value = [turb_phi]
        
        phi_value = np.asarray(phi_value).flatten()
        
        # Extract 1D arrays from turb_s2 and turb_s2_max DataFrames
        turb_s2_values = turb_s2.iloc[:, 1].values if hasattr(turb_s2, 'iloc') else np.asarray(turb_s2).flatten()
        turb_s2_max_values = turb_s2_max.iloc[:, 1].values if hasattr(turb_s2_max, 'iloc') else np.asarray(turb_s2_max).flatten()
        
        tmp = pd.DataFrame({'phi': phi_value, 's2_tx': turb_s2_values, 's2max': turb_s2_max_values})
        maxT = tmp.apply(lambda row: turb([row['phi'], row['s2_tx'], row['s2max']]), axis=1).to_numpy()

        Tx_zero = np.where(Tx < 1)[0]
        Tx = (Tx - 1) / (maxT - 1)
        if len(Tx_zero) > 0:
            Tx[Tx_zero, :] = 0

    Tx_df = pd.DataFrame(Tx, index=seqdata.seqdata.index, columns=['Turbulence'])
    
    # Handle ID display options
    if id_as_column:
        # Add ID as a separate column and reset index to numeric
        Tx_df['ID'] = Tx_df.index
        Tx_df = Tx_df[['ID', 'Turbulence']].reset_index(drop=True)
    else:
        # Always set index name to 'ID' for clarity
        Tx_df.index.name = 'ID'
    
    return Tx_df