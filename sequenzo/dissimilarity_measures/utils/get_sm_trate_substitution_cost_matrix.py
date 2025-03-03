"""
@Author  : 李欣怡
@File    : get_sm_trate_substitution_cost_matrix.py
@Time    : 2024/11/25 13:20
@Desc    : Computes transition rates
"""

import numpy as np
from sequenzo.define_sequence_data import SequenceData

def get_sm_trate_cost_matrix(seqdata, time_varying=False, weighted=True, lag=1, count=False):
    # ================
    # Check Parameters
    # ================
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[x] Seqdata must be a pandas DataFrame representing sequences.")

    # Weights
    if weighted:
        weights = seqdata.weights
    else:
        weights = np.ones(seqdata.seqdata.shape[0])

    states = seqdata.states
    statesMapping = seqdata.state_mapping

    _size = len(states) + 1
    sdur = seqdata.seqdata.shape[1]

    if lag < 0:
        all_transition = np.arange(abs(lag), sdur)
    else:
        all_transition = np.arange(sdur - lag)

    num_transition = len(all_transition)

    # =====================================
    # Compute Time Varying Transition Rates
    # =====================================
    seqdata = seqdata.seqdata
    if time_varying:
        """
        >>> for example:
                , , C1
                                             [-> Non computing] [-> Non technical computing] [-> Technical computing]
                [Non computing ->]                   0.81126761                  0.014084507                0.1746479
                [Non technical computing ->]         0.08571429                  0.800000000                0.1142857
                [Technical computing ->]             0.04098361                  0.004918033                0.9540984

                , , C2
                                             [-> Non computing] [-> Non technical computing] [-> Technical computing]
                [Non computing ->]                   0.87658228                  0.003164557               0.12025316
                [Non technical computing ->]         0.05555556                  0.916666667               0.02777778
                [Technical computing ->]             0.06481481                  0.009259259               0.92592593

                , , C3
                                             [-> Non computing] [-> Non technical computing] [-> Technical computing]
                [Non computing ->]                   0.90654206                  0.003115265               0.09034268
                [Non technical computing ->]         0.05000000                  0.850000000               0.10000000
                [Technical computing ->]             0.06729264                  0.006259781               0.92644757

                , , C4
                                             [-> Non computing] [-> Non technical computing] [-> Technical computing]
                [Non computing ->]                   0.87500000                  0.005952381               0.11904762
                [Non technical computing ->]         0.07692308                  0.846153846               0.07692308
                [Technical computing ->]             0.05760000                  0.012800000               0.92960000
        """

        tmat = np.zeros((num_transition, _size, _size))

        for sl in all_transition:
            missing_cond = np.not_equal(seqdata.iloc[:, sl + lag], np.nan)

            for state_x in statesMapping.values():
                colx_cond = np.equal(seqdata.iloc[:, sl], state_x)
                PA = np.sum(weights[colx_cond & missing_cond])

                if PA == 0:
                    tmat[sl, state_x, :] = 0
                else:
                    for state_y in statesMapping.values():
                        PAB = np.sum(weights[colx_cond & (np.equal(seqdata.iloc[:, sl + lag], state_y))])
                        tmat[sl, state_x, state_y] = PAB if count else PAB / PA

    # =========================================
    # Compute Non Time Varying Transition Rates
    # =========================================
    else:
        seqdata = seqdata.to_numpy()

        tmat = np.zeros((_size, _size))

        missing_cond = np.not_equal(seqdata[:, all_transition + lag], np.nan)

        for state_x in statesMapping.values():
            PA = 0
            colx_cond = np.equal(seqdata[:, all_transition], state_x)

            if num_transition > 1:
                PA = np.sum(weights * np.sum(colx_cond & missing_cond, axis=1))
            else:
                PA = np.sum(weights * (colx_cond & missing_cond))

            if PA == 0:
                tmat[state_x, :] = 0
            else:
                for state_y in statesMapping.values():
                    if num_transition > 1:
                        PAB = np.sum(weights *
                                     np.sum(colx_cond & (np.equal(seqdata[:, all_transition + lag], state_y)),
                                            axis=1))
                    else:
                        PAB = np.sum(
                            weights * (colx_cond & (np.equal(seqdata[:, all_transition + lag], state_y))))

                    tmat[state_x, state_y] = PAB if count else PAB / PA

    return tmat