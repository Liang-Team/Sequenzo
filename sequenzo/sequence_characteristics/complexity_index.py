"""
@Author  : 李欣怡
@File    : complexity_index.py
@Time    : 2025/9/23 23:45
@Desc    : Complexity index

        The corresponding function name in TraMineR is seqici.R,
        with the source code available at: https://github.com/cran/TraMineR/blob/master/R/seqici.R
"""
import os
from contextlib import redirect_stdout
import pandas as pd
import numpy as np

from sequenzo.define_sequence_data import SequenceData
from .simple_characteristics import get_number_of_transitions
from .within_sequence_entropy import get_within_sequence_entropy

def get_complexity_index(seqdata, silent=True):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData function to create one.")

    if not silent:
        print(f"[>] Computing complexity index for {seqdata.seqdata.shape[0]} sequences ...")

    trans = get_number_of_transitions(seqdata=seqdata, norm=True)

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            ient = get_within_sequence_entropy(seqdata=seqdata, norm=True)

    complxity = np.sqrt(trans.iloc[:, 0] * ient.iloc[:, 0])
    complxity = pd.DataFrame(complxity, index=seqdata.seqdata.index, columns=['ComplexityIndex'])

    return complxity