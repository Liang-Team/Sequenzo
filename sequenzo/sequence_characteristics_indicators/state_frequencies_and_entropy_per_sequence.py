"""
@Author  : 李欣怡
@File    : state_frequencies_and_entropy_per_sequence.py
@Time    : 2025/9/23 19:34
@Desc    : State distribution for each individual

        The corresponding function name in TraMineR is seqistatd.R,
        with the source code available at: https://github.com/cran/TraMineR/blob/master/R/seqistatd.R
"""
import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData

def get_state_freq_and_entropy_per_seq(seqdata, prop=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData function to create one.")

    if seqdata.labels is not None:
        states = seqdata.labels
    else:
        states = seqdata.states

    number_states = len(states)
    number_seq = seqdata.seqdata.shape[0]

    iseqtab = pd.DataFrame(np.zeros((number_seq, number_states)), index=seqdata.seqdata.index, columns=states)

    print(f"[>] Computing state distribution for {number_seq} sequences and {number_states} states ...")

    for i, state in enumerate(states):
        iseqtab.iloc[:, i] = (seqdata.seqdata == (i+1)).sum(axis=1)

    if prop:
        iseqtab = iseqtab.div(iseqtab.sum(axis=1), axis=0)

    iseqtab = iseqtab.reset_index().rename(columns={'index': 'ID'})

    return iseqtab