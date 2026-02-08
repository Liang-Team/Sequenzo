"""
@Author  : 李欣怡
@File    : get_substitution_cost_matrix.py
@Time    : 2024/11/11 12:00
@Desc    : Compute substitution costs and substitution-cost/proximity matrix
"""
import warnings

import pandas as pd
import numpy as np

from .utils.get_sm_trate_substitution_cost_matrix import get_sm_trate_substitution_cost_matrix
from sequenzo.define_sequence_data import SequenceData
from sequenzo.sequence_characteristics.overall_cross_sectional_entropy import get_cross_sectional_entropy
from .get_distance_matrix import with_missing_warned

def get_substitution_cost_matrix(seqdata, method, cval=None, miss_cost=None, time_varying=False,
                                 weighted=True, transition="both", lag=1, miss_cost_fixed=None,
                                 **kwargs):
    if 'with_missing' in kwargs and not with_missing_warned:
        print("[!] 'with_missing' has been removed and is ignored.")
        print("    Missing values are always included by default, consistent with TraMineR.")

    # ================
    # Check Parameters
    # ================
    if not isinstance(seqdata, SequenceData):
        raise ValueError(" [!] data is NOT a sequence object, see SequenceData function to create one.")

    metlist = ["CONSTANT", "TRATE", "INDELS", "INDELSLOG", "FUTURE", "FEATURES"]
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

    # ==================
    # Process "FUTURE"
    # ==================
    # TraMineR: substitution costs from chi-squared distance between rows of the
    # transition rate matrix (common future). Not time-varying.
    if method == "FUTURE":
        if time_varying:
            raise ValueError("[!] time.varying substitution cost is not (yet) implemented for method FUTURE.")
        print("[>] Creating substitution-cost matrix using common future...")
        tr = get_sm_trate_substitution_cost_matrix(
            seqdata, time_varying=False, weighted=weighted, lag=lag
        )
        # TraMineR: with.missing=FALSE -> alphabet has len(states) only; chisqdista on 6x6 transition block.
        # Our tr is (alphsize, alphsize) with row/col 0 unused; use only state block [1:alphsize, 1:alphsize] -> nstates x nstates.
        n_states = len(states)
        tr_states = tr[1 : n_states + 1, 1 : n_states + 1]
        cs = np.sum(tr_states, axis=0)
        pdot = np.zeros_like(cs)
        np.place(pdot, cs > 0, 1.0 / np.where(cs > 0, cs, 1.0))
        costs_ss = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(i + 1, n_states):
                d = np.sum(pdot * (tr_states[i, :] - tr_states[j, :]) ** 2)
                costs_ss[i, j] = np.sqrt(d)
                costs_ss[j, i] = costs_ss[i, j]
        np.fill_diagonal(costs_ss, 0)
        max_ss = np.max(costs_ss)
        return_result["indel"] = 0.5 * max_ss
        # Embed in alphsize x alphsize for downstream (row/col 0 = null state)
        costs = np.zeros((alphsize, alphsize), dtype=np.float64)
        costs[1 : n_states + 1, 1 : n_states + 1] = costs_ss
        costs[0, 1 : n_states + 1] = max_ss
        costs[1 : n_states + 1, 0] = max_ss
        costs[0, 0] = 0.0

    # ==================
    # Process "FEATURES"
    # ==================
    # TraMineR: substitution costs from Gower distance on state features (cluster::daisy).
    # state_features: one row per state (same order as alphabet); optional weights and types.
    if method == "FEATURES":
        if time_varying:
            raise ValueError("[!] time.varying substitution cost is not (yet) implemented for method FEATURES.")
        state_features = kwargs.get("state_features")
        feature_weights = kwargs.get("feature_weights", None)
        feature_type = kwargs.get("feature_type", None)
        if state_features is None or not isinstance(state_features, (pd.DataFrame, np.ndarray)):
            raise ValueError(
                "[!] state_features should be a DataFrame or array with one row per state (and optionally one for missing)."
            )
        if hasattr(state_features, "values"):
            X = np.asarray(state_features.values, dtype=np.float64)
        else:
            X = np.asarray(state_features, dtype=np.float64)
        if X.shape[0] not in (len(states), alphsize):
            raise ValueError(
                f"[!] state_features must have {len(states)} or {alphsize} rows (one per state, optionally one for missing), got {X.shape[0]}."
            )
        # TraMineR: state.features has one row per state (6 rows); daisy gives 6x6. We build 6x6 then embed in 7x7.
        use_embed = X.shape[0] == len(states)
        n_ss = len(states) if use_embed else alphsize
        X_ss = X[:n_ss]
        if feature_weights is None:
            feature_weights = np.ones(X_ss.shape[1], dtype=np.float64)
        else:
            feature_weights = np.asarray(feature_weights, dtype=np.float64)
            if feature_weights.size != X_ss.shape[1]:
                feature_weights = np.resize(
                    np.atleast_1d(feature_weights), X_ss.shape[1]
                )
        ranges = np.nanmax(X_ss, axis=0) - np.nanmin(X_ss, axis=0)
        ranges[ranges == 0] = 1.0
        costs_ss = np.zeros((n_ss, n_ss))
        for i in range(n_ss):
            for j in range(i + 1, n_ss):
                d = np.nansum(
                    feature_weights * np.abs(X_ss[i, :] - X_ss[j, :]) / ranges
                ) / np.sum(feature_weights)
                costs_ss[i, j] = d
                costs_ss[j, i] = d
        np.fill_diagonal(costs_ss, 0)
        max_ss = np.max(costs_ss)
        return_result["indel"] = 0.5 * max_ss
        if use_embed:
            costs = np.zeros((alphsize, alphsize), dtype=np.float64)
            costs[1 : n_ss + 1, 1 : n_ss + 1] = costs_ss
            costs[0, 1 : n_ss + 1] = max_ss
            costs[1 : n_ss + 1, 0] = max_ss
            costs[0, 0] = 0.0
        else:
            costs = np.asarray(costs_ss, dtype=np.float64)

    # ================================
    # Process "INDELS" and "INDELSLOG"
    # ================================
    if method in ["INDELS", "INDELSLOG"]:
        if time_varying:
            indels = get_cross_sectional_entropy(seqdata, return_format="dict")['Frequencies']
        else:
            ww = seqdata.weights
            if ww is None:
                ww = np.ones(seqdata.seqdata.shape[0])

            flat_seq = seqdata.values.flatten(order='F')
            weights_rep = np.repeat(ww, seqdata.seqdata.shape[1])
            df = pd.DataFrame({'state': flat_seq, 'weight': weights_rep})
            weighted_counts = df.groupby('state')['weight'].sum()

            weighted_prob = weighted_counts / weighted_counts.sum()
            states_num = range(1, len(seqdata.states) + 1)
            indels = np.array([weighted_prob.get(s, 0) for s in states_num])

        indels[np.isnan(indels)] = 1
        if method == "INDELSLOG":
            indels = np.log(2 / (1 + indels))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                indels = 1 / indels
            indels[np.isinf(indels)] = 1e15  # 避免cast警告

        if time_varying:
            return_result['indel'] = indels
        else:
            return_result['indel'] = np.insert(indels, 0, 0)    # cause C++ is 1-indexed

        if time_varying:
            time = seqdata.seqdata.shape[1]

            print(
                f"  - Creating {alphsize}x{alphsize}x{time} time varying substitution-cost matrix using {cval} as constant value.")
            costs = np.full((time, alphsize, alphsize), 0.0)

            for t in range(time):
                for i in range(1, alphsize):
                    for j in range(1, alphsize):
                        if i != j:
                            val = indels.iloc[i - 1, t] + indels.iloc[j - 1, t]
                            costs[t, i, j] = np.clip(val, -1e15, 1e15)  # 避免cast警告

        else:
            costs = np.full((alphsize, alphsize), 0.0)
            for i in range(1, alphsize):
                for j in range(1, alphsize):
                    if i != j:
                        costs[i, j] = indels[i - 1] + indels[j - 1]
            costs[np.isinf(costs)] = 1e15  # 避免cast警告

    # =================================
    # Process the Cost of Missing Value
    # =================================
    if seqdata.ismissing and miss_cost_fixed:
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
    from sequenzo.dissimilarity_measures import get_substitution_cost_matrix

    df = pd.read_csv('/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/country_co2_emissions_missing.csv')

    time = list(df.columns)[1:]

    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    sequence_data = SequenceData(df, time=time, id_col="country", states=states)

    sm = get_substitution_cost_matrix(sequence_data,
                                      method="CONSTANT",
                                      cval=2,
                                      time_varying=False)

    print("===============")