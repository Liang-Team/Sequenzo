"""
@Author  : Yuqi Liang 梁彧祺
@File    : linked_polyad.py
@Time    : 28/04/2025 21:19
@Desc    :
    This module implements the full Python version of Tim Liao and Gilbert Ritschard's
    seqpolyads function (R version 1.0.2, 29.12.20) for linked polyadic sequence analysis.

    Provided functionalities:
    1. Customizable pairwise weighting (pair_weights)
    2. Support for role-specific weights (role_weights)
    3. Support for weighted sampling (weights)
    4. Randomization method selection: a=1 (resample sequences), a=2 (resample states)
    5. Multi-core parallel processing (n_jobs)
    6. Full reproducibility via random_seed control
    7. Outputs include observed distances, randomized distances, U, V, V>0.95 dummy, and mean observed/random distances

    All calculations faithfully replicate the logic and outputs of the original R implementation.

    Note:
    You may encounter the following error during execution, especially when running the script inside PyCharm:

    Traceback (most recent call last):
      File "/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/_pydevd_bundle/pydevd_comm.py", line 293, in _on_run
        r = self.sock.recv(1024)
    OSError: [Errno 9] Bad file descriptor
    This error is related to PyCharm's debugger trying to manage communication sockets while multiprocessing or background progress bars (like tqdm) are active.
    It does not affect the actual computation or results of the linked_polyad function. You can safely ignore it.

    To suppress it or avoid seeing it:

    Run the script outside the PyCharm debugger (e.g., from terminal or using “Run” instead of “Debug”).

    Alternatively, disable progress bars or multiprocessing (e.g., set n_jobs=1 and disable=True in tqdm, if available in the function).
"""
import numpy as np
import random
from typing import List, Dict, Union
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.define_sequence_data import SequenceData

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import multiprocessing
import platform

if platform.system() != "Windows":
    multiprocessing.set_start_method("fork", force=True)


def linked_polyadic_sequence_analysis(seqlist: List[SequenceData],
                                      a: int = 1,
                                      method: str = "OM",
                                      distance_parameters: dict = None,
                                      weights: np.ndarray = None,
                                      rand_weight_type: int = 1,
                                      role_weights: List[float] = None,
                                      pair_weights: np.ndarray = None,
                                      T: int = 1000,
                                      random_seed: int = 36963,
                                      replace: bool = True,
                                      n_jobs: int = 1,
                                      verbose: bool = True,
                                      return_df: bool = False) -> Union[Dict, pd.DataFrame]:
    """
    Calculate U and V statistics for linked polyadic sequence data.

    Provided functionalities:
    1. Customizable pairwise weighting (pair_weights)
    2. Support for role-specific weights (role_weights)
    3. Support for weighted sampling (weights)
    4. Randomization method selection: a=1 (resample sequences), a=2 (resample states)
    5. Multi-core parallel processing (n_jobs)
    6. Full reproducibility via random_seed control
    7. Outputs include observed distances, randomized distances, U, V, V>0.95 dummy, and mean observed/random distances

    :param seqlist: List of SequenceData objects to analyze.
    :param a: Randomization type. 1 = resample sequences; 2 = resample states within sequences.
    :param method: Distance measure method ('HAM', 'OM', 'CHI2', etc.).
    :param distance_parameters: Dictionary of additional keyword arguments for distance calculation.
    :param weights: Sampling weights for sequences when generating random polyads.
    :param rand_weight_type: Strategy for computing randomization weights (1 = uniform, 2 = sample-weight-based).
    :param role_weights: Role-specific importance weights for different sequence sources.
    :param pair_weights: Pairwise weights for distance averaging.
    :param T: Number of randomizations performed.
    :param random_seed: Seed for random number generator to ensure reproducibility.
    :param replace: Whether to sample with replacement during randomization.
    :param n_jobs: Number of parallel workers for randomization; set to -1 to use all CPUs.
    :param verbose: Whether to display a progress bar during randomization.
    :param return_df: If True, return formatted DataFrame instead of dictionary.

    Dictionary containing:
    - 'mean.dist': Mean observed and random distances
    - 'U': Array of U values (mean random distance - observed distance)
    - 'V': Array of V values (proportion observed < random)
    - 'V.95': Binary array where V > 0.95
    - 'observed.dist': Array of observed polyadic distances
    - 'random.dist': Array of randomized polyadic distances
    """
    if distance_parameters is None:
        distance_parameters = {}

    P = len(seqlist)
    n = seqlist[0].n_sequences
    seq_length = seqlist[0].n_steps

    for sd in seqlist:
        assert isinstance(sd, SequenceData)
        assert sd.n_sequences == n
        assert sd.n_steps == seq_length

    if role_weights is None:
        role_weights = [1.0 / P] * P
    role_weights = np.array(role_weights)

    if pair_weights is None:
        pair_weights = np.ones(int(P * (P - 1) / 2))

    tagged_dfs = []
    for i, sd in enumerate(seqlist):
        df = sd.to_dataframe().copy()
        df["__id__"] = [f"R{i}_{j}" for j in range(sd.n_sequences)]
        tagged_dfs.append(df)

    data_concat = pd.concat(tagged_dfs, axis=0).reset_index(drop=True)
    merged_seqdata = SequenceData(
        data=data_concat,
        time=seqlist[0].time,
        time_type="age",
        states=[i for i in range(1, len(seqlist[0].states) + 1)],
        id_col="__id__"
    )

    alldist = np.asarray(get_distance_matrix(merged_seqdata, method=method, **distance_parameters))
    cj = np.array([n * p for p in range(P)])

    if weights is None:
        weights = np.ones(n) / n

    def weighted_mean(mat):
        return np.average(mat[np.triu_indices(P, 1)], weights=pair_weights)

    l_m = np.zeros((T, P), dtype=int)

    def random_sample_once(i):
        local_rng = np.random.default_rng(random_seed + i)
        sampled = local_rng.choice(n, size=P, replace=replace, p=weights)
        l_m[i] = sampled
        sample_indices = cj + sampled

        if a == 1:
            mat = alldist[np.ix_(sample_indices, sample_indices)]
            return weighted_mean(mat)
        elif a == 2:
            df = merged_seqdata.to_dataframe().drop(columns="__id__")
            sampled_df = df.iloc[sample_indices].reset_index(drop=True)
            shuffled = sampled_df.apply(lambda row: local_rng.choice(row, size=seq_length, replace=replace),
                                        axis=1, result_type="broadcast")
            shuffled["__id__"] = [f"Rand_{i}_{j}" for j in range(len(shuffled))]
            seq_shuffled = SequenceData(
                data=shuffled,
                time=merged_seqdata.time,
                time_type=merged_seqdata.time_type,
                states=merged_seqdata.states,
                id_col="__id__"
            )
            dmat = np.asarray(get_distance_matrix(seq_shuffled, method=method, **distance_parameters))
            return weighted_mean(dmat)
        else:
            raise ValueError("Invalid randomization type 'a'. Should be 1 or 2.")

    iterator = tqdm(range(T), desc="Randomizing polyads") if verbose else range(T)
    random_dists = Parallel(n_jobs=n_jobs)(delayed(random_sample_once)(i) for i in iterator)
    random_dists = np.array(random_dists)

    observed_dists = []
    for i in range(n):
        indices = [i + n * p for p in range(P)]
        mat = alldist[np.ix_(indices, indices)]
        observed_dists.append(weighted_mean(mat))
    observed_dists = np.array(observed_dists)

    if rand_weight_type == 2:
        p_weights = np.array([np.sum(weights[sampled]) for sampled in l_m])
    else:
        p_weights = 1.0

    l_weights = np.zeros(T)
    for i in range(T):
        sampled = l_m[i]
        l_weights[i] = np.sum(weights[sampled] * role_weights / p_weights[i] if rand_weight_type == 2 else p_weights)
    l_weights /= np.sum(l_weights)

    mean_rand_dist = np.sum(random_dists * l_weights)
    U = mean_rand_dist - observed_dists
    V = np.array([np.sum((observed_dists[i] < random_dists) * l_weights) for i in range(n)])
    V_95 = (V > 0.95).astype(int)

    result = {
        "mean.dist": {"Obs": np.mean(observed_dists), "Rand": mean_rand_dist},
        "U": U,
        "V": V,
        "V.95": V_95,
        "observed.dist": observed_dists,
        "random.dist": random_dists
    }

    if return_df:
        return pd.DataFrame({
            'ObservedDist': result['observed.dist'],
            'U': result['U'],
            'V': result['V'],
            'V>0.95': result['V.95']
        }, index=pd.RangeIndex(start=1, stop=len(result['U']) + 1, name="PolyadID"))
    else:
        return result


if __name__ == '__main__':
    pass
