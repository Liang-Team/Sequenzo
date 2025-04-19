"""
@Author  : 李欣怡
@File    : get_substitution_cost_matrix.py
@Time    : 2024/11/11 12:00
@Desc    : Compute substitution costs and substitution-cost/proximity matrix
"""

import pandas as pd
import numpy as np
from .utils.get_sm_trate_substitution_cost_matrix import get_sm_trate_substitution_cost_matrix
from sequenzo.define_sequence_data import SequenceData


def get_substitution_cost_matrix(seqdata, method, cval=None, with_missing=False, miss_cost=None,
                                 time_varying=False, weighted=True, transition="both", lag=1,
                                 miss_cost_fixed=None):
    # ================
    # Check Parameters
    # ================
    if not isinstance(seqdata, SequenceData):
        raise ValueError(" [!] data is NOT a sequence object, see SequenceData function to create one.")

    metlist = ["CONSTANT", "TRATE"]
    if method not in metlist:
        raise ValueError(f" [!] method must be one of: {', '.join(metlist)}.")

    transitionlist = ["previous", "next", "both"]
    if transition not in transitionlist:
        raise ValueError(f" [!] transition must be one of: {', '.join(transitionlist)}.")

    return_result = {"indel": 1}

    cval4cond = time_varying and method == "TRATE" and transition == "both"
    if cval is None:
        cval = 4 if cval4cond else 2
    if miss_cost is None:
        miss_cost = cval
    if miss_cost_fixed is None:
        miss_cost_fixed = False if method in ["INDELS", "INDELSLOG"] else True

    states = seqdata.states.copy()
    alphsize = len(states) + 1

    # ==================
    # Process "CONSTANT"
    # ==================
    if method == "CONSTANT":
        if cval is None:
            raise ValueError("[!] No value for the constant substitution-cost.")

        if time_varying:
            time = seqdata.seqdata.shape[1]

            print(
                f"  - Creating {alphsize}x{alphsize}x{time} time varying substitution-cost matrix using {cval} as constant value.")
            costs = np.full((time, alphsize, alphsize), cval)

            for i in range(time):
                np.fill_diagonal(costs[i, :, :], 0)  # Set diagonal to 0 in each time slice
        else:
            print(f"  - Creating {alphsize}x{alphsize} substitution-cost matrix using {cval} as constant value")
            costs = np.full((alphsize, alphsize), cval)
            np.fill_diagonal(costs, 0)  # Set diagonal to 0

    # ===============
    # Process "TRATE"
    # ===============
    if method == "TRATE":
        print("[>] Transition-based substitution-cost matrix (TRATE) initiated...")
        print(f"  - Computing transition probabilities for: [{', '.join(map(str, seqdata.states))}]")   # Because the matrix CLARA is passing in is a number

        if time_varying:
            tr = get_sm_trate_substitution_cost_matrix(seqdata, time_varying=True, weighted=weighted, lag=lag)

            tmat = tr.shape[1]               # Number of states (since tr is three dimensions np.ndarray, the first dimension is time)
            time = seqdata.seqdata.shape[1]  # Total number of time points
            costs = np.zeros((time, alphsize, alphsize))

            # Function to compute the cost according to transition rates
            def tratecostBoth(trate, t, state1, state2, debut, fin):
                cost = 0
                if not debut:
                    # the first state
                    cost -= trate[t - 1, state1, state2] + trate[t - 1, state2, state1]
                if not fin:
                    # the last state
                    cost -= trate[t, state1, state2] + trate[t, state2, state1]
                return cost + cval if not debut and not fin else cval + 2 * cost

            def tratecostPrevious(trate, t, state1, state2, debut, fin):
                cost = 0
                if not debut:
                    # the first state
                    cost -= trate[t - 1, state1, state2] + trate[t - 1, state2, state1]
                return cval + cost

            def tratecostNext(trate, t, state1, state2, debut, fin):
                cost = 0
                if not fin:
                    # the last state
                    cost -= trate[t, state1, state2] + trate[t, state2, state1]
                return cval + cost

            if transition == "previous":
                tratecost = tratecostPrevious
            elif transition == "next":
                tratecost = tratecostNext
            else:
                tratecost = tratecostBoth

            for t in range(time):
                for i in range(tmat - 1):
                    for j in range(i + 1, tmat):
                        cost = max(0, tratecost(tr, t, i, j, debut=(t == 0), fin=(t == time - 1)))
                        costs[t, i, j] = cost
                        costs[t, j, i] = cost

        else:
            tr = get_sm_trate_substitution_cost_matrix(seqdata, time_varying=False, weighted=weighted, lag=lag)

            tmat = tr.shape[0]
            costs = np.zeros((alphsize, alphsize))

            for i in range(1, tmat - 1):
                for j in range(i + 1, tmat):
                    cost = cval - tr[i, j] - tr[j, i]
                    costs[i, j] = cost
                    costs[j, i] = cost

            indel = 0.5 * np.max(costs)

            return_result['indel'] = indel

    print("[>] Indel cost generated.\n")

    # =================================
    # Process the Cost of Missing Value
    # =================================
    if with_missing and miss_cost_fixed:
        if time_varying:
            costs[:, alphsize - 1, :alphsize - 1] = miss_cost
            costs[:, :alphsize - 1, alphsize - 1] = miss_cost
        else:
            costs[alphsize - 1, :alphsize - 1] = miss_cost
            costs[:alphsize - 1, alphsize - 1] = miss_cost

    # ===============================
    # Setting Rows and Columns Labels
    # ===============================
    if time_varying:    # 3D
        costs = costs
    else:   # 2D
        states.insert(0, "null")
        costs = pd.DataFrame(costs, index=states, columns=states, dtype=float)

    # ===============================
    # Calculate the Similarity Matrix
    # ===============================
    return_result['sm'] = costs

    return return_result


# Define seqsubm as an alias for backward compatibility
def seqsubm(*args, **kwargs):
    return get_substitution_cost_matrix(*args, **kwargs)['sm']

if __name__ == "__main__":
    df = pd.read_csv('D:/country_co2_emissions_missing.csv')

    time = list(df.columns)[1:]

    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    sequence_data = SequenceData(df, time=time, time_type="year", id_col="country", states=states)

    sm = get_substitution_cost_matrix(sequence_data,
                                      method="CONSTANT",
                                      cval=2,
                                      time_varying=False)

    print("===============")