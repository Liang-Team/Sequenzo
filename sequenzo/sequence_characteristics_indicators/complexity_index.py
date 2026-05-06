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

    trans = get_number_of_transitions(seqdata=seqdata, norm=True).iloc[:, 1]
    trans.index = seqdata.seqdata.index

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            ient = get_within_sequence_entropy(seqdata=seqdata, norm=True).iloc[:, 1]
            ient.index = seqdata.seqdata.index

    complxity = np.sqrt(trans * ient)
    complxity = pd.DataFrame(complxity, index=seqdata.seqdata.index, columns=['Complexity Index'])
    complxity = complxity.reset_index().rename(columns={'index': 'ID'})

    return complxity

if __name__ == '__main__':
    from sequenzo import *

    df = load_dataset("country_co2_emissions")
    _time = list(df.columns)[1:]
    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    sequence_data = SequenceData(df, time=_time, id_col="country", states=states)

    res = get_complexity_index(sequence_data)
    res