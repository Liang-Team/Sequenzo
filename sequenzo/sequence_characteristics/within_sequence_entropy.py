"""
@Author  : 李欣怡
@File    : within_sequence_entropy.py
@Time    : 2025/9/23 19:44
@Desc    : Within Sequence Entropy

        The corresponding function name in TraMineR is seqient.R,
        with the source code available at: https://github.com/cran/TraMineR/blob/master/R/seqient.R
"""
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.stats import entropy

from sequenzo.define_sequence_data import SequenceData
from .state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq

def get_within_sequence_entropy(seqdata, norm=True, base=np.e, silent=True):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData function to create one.")

    states = seqdata.states.copy()

    if not silent:
        print(f"  - computing within sequence entropy for {seqdata.seqdata.shape[0]} sequences and {len(states)} states ...")

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            iseqtab = get_state_freq_and_entropy_per_seq(seqdata=seqdata)
            iseqtab.index = seqdata.seqdata.index

    ient = iseqtab.iloc[:, 1:].apply(lambda row: entropy(row, base=base), axis=1)

    if norm:
        maxent = np.log(len(states))
        ient = ient / maxent

    ient = pd.DataFrame(ient, index=seqdata.seqdata.index, columns=['Entropy'])
    ient = ient.reset_index().rename(columns={'index': 'ID'})

    return ient
