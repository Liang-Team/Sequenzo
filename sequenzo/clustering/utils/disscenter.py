"""
@Author  : 李欣怡
@File    : disscenter.py
@Time    : 2025/2/8 12:57
@Desc    :
    Utility function for the k_medoids algorithm.
"""

import numpy as np
import pandas as pd

import importlib
import sequenzo.dissimilarity_measures.c_code

c_code = importlib.import_module("sequenzo.dissimilarity_measures.c_code")

def disscentertrim(diss, group=None, medoids_index=None, allcenter=False, weights=None, squared=False, trim=0):

    # Lazily import the c_code module to avoid circular dependencies during installation
    # from .__init__ import _import_c_code
    # c_code = _import_c_code()

    if isinstance(medoids_index, bool):
        if medoids_index:
            medoids_index = "First"
        else:
            medoids_index = None

    retmedoids = medoids_index is not None  # Whether medoids need to be returned
    if retmedoids:
        allcenter = False

    allmedoids = False

    if medoids_index is not None:
        if medoids_index == "all":
            allmedoids = True
        elif medoids_index != "first":
            raise ValueError('\'medoids_index\' argument should be one of "First", "all" or None')

    if weights is None:
        weights = np.ones(len(diss), dtype=float)

    if squared:
        diss = np.square(diss)

    if group is None:
        group = np.ones(diss.shape[0], dtype=int)

    ind = np.arange(diss.shape[0])
    grp = np.array(group)
    lgrp = np.unique(group)

    if allcenter:
        ret = pd.DataFrame(np.zeros((diss.shape[0], 1)))
    else:
        ret = np.zeros(diss.shape[0])

    if retmedoids:
        if allmedoids:
            medoids = []
        else:
            medoids = np.zeros(len(lgrp))

    for i in range(len(lgrp)):
        cond = (grp == lgrp[i])
        grpindiv = ind[cond]   # 第 i 组所有数据点在隶属矩阵里的位置（0-based 索引）

        if allcenter:
            # TODO : 以后再补充
            print("以后再补充")

        else:
            inertia = c_code.weightedinertia(diss.astype(np.float64),
                                             grpindiv.astype(np.int32),
                                             weights.astype(np.float64))
            dc = inertia.tmrWeightedInertiaContrib()
            dc = dc - np.average(dc, weights=weights[cond]) / 2

            if trim > 0:
                # TODO : 以后再补充
                print("以后再补充")

            ret[grpindiv] = dc
            mindc = np.min(dc)

            if retmedoids:
                if allmedoids:
                    medoids.append(np.where((ret == mindc) & cond)[0])
                else:
                    medoids[i] = np.where((ret == mindc) & cond)[0][0]

    if retmedoids:
        if len(lgrp) == 1:
            return medoids[[1]]

        return medoids

    return ret


if __name__ == '__main__':
    # Load the data that we would like to explore in this tutorial
    # `df` is the short for `dataframe`, which is a common variable name for a dataset
    from sequenzo import *
    df = load_dataset('country_co2_emissions')

    time = list(df.columns)[1:]

    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    sequence_data = SequenceData(df, time=time, states=states)

    result = clara(seqdata=sequence_data,
                   R=2,
                   kvals=range(2, 21),
                   sample_size=3000,
                   criteria=['distance', 'pbm'],
                   parallel=True,
                   stability=True)
    result = result['allstat']

