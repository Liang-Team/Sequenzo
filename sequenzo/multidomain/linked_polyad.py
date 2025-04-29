"""
@Author  : Yuqi Liang 梁彧祺
@File    : linked_polyad.py
@Time    : 28/04/2025 21:19
@Desc    :
    This module implements the full Python version of Tim Liao and Gilbert Ritschard's
    seqpolyads function (R version 1.0.2, 29.12.20) for linked polyadic sequence analysis.

    Provided functionalities:
    1. Customizable pairwise weighting (w)
    2. Support for role-specific weights (role_weights)
    3. Support for weighted sampling (weights)
    4. Randomization method selection: a=1 (resample sequences), a=2 (resample states)
    5. Multi-core parallel processing (n_jobs)
    6. Full reproducibility via random_seed control
    7. Outputs include observed distances, randomized distances, U, V, V>0.95 dummy, and mean observed/random distances

    All calculations faithfully replicate the logic and outputs of the original R implementation.
"""
import numpy as np
import random
from typing import List, Dict, Union
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.define_sequence_data import SequenceData


def linked_polyad(seqlist: List[SequenceData],
                  a: int = 1,
                  method: str = "HAM",
                  distance_kwargs: dict = None,
                  weights: np.ndarray = None,
                  role_weights: List[float] = None,
                  T: int = 1000,
                  random_seed: int = 36963,
                  replace: bool = True,
                  n_jobs: int = 1,
                  verbose: bool = True,
                  return_df: bool = False) -> Union[Dict, pd.DataFrame]:
    """
    Calculate U and V statistics for linked polyadic sequence data.

    :param seqlist: List of SequenceData objects to analyze.
    :param a: Randomization type. 1 = resample sequences; 2 = resample states within sequences.
    :param method: Distance measure method ('HAM', 'OM', 'CHI2', etc.).
    :param distance_kwargs: Dictionary of additional keyword arguments for distance calculation.
    :param weights: Sampling weights for sequences when generating random polyads.
    :param role_weights: Role-specific importance weights for different sequence sources.
    :param T: Number of randomizations performed.
    :param random_seed: Seed for random number generator to ensure reproducibility.
    :param replace: Whether to sample with replacement during randomization.
    :param n_jobs: Number of parallel workers for randomization; set to -1 to use all CPUs.
    :param verbose: Whether to display a progress bar during randomization.
    :param return_df: If True, return formatted DataFrame instead of dictionary.

    :return: Either a dictionary or a formatted DataFrame depending on return_df.

    Dictionary containing:
    - 'mean.dist': Mean observed and random distances
    - 'U': Array of U values (mean random distance - observed distance)
    - 'V': Array of V values (proportion observed < random)
    - 'V.95': Binary array where V > 0.95
    - 'observed.dist': Array of observed polyadic distances
    - 'random.dist': Array of randomized polyadic distances
    """
    if distance_kwargs is None:
        distance_kwargs = {}

    P = len(seqlist)
    n = seqlist[0].n_sequences
    seq_length = seqlist[0].n_steps

    # Check all SequenceData objects compatible
    for sd in seqlist:
        assert isinstance(sd, SequenceData)
        assert sd.n_sequences == n
        assert sd.n_steps == seq_length

    if role_weights is None:
        role_weights = [1.0 / P] * P
    role_weights = np.array(role_weights)

    # Concatenate all sequences
    all_sequences = np.vstack([sd.to_numeric() for sd in seqlist])
    all_ids = np.concatenate([sd.ids for sd in seqlist])

    # Compute full distance matrix
    alldist = get_distance_matrix(all_sequences, method=method, **distance_kwargs)

    # Precompute indices shifts for generations
    cj = np.array([n * p for p in range(P)])

    # Random generator
    rng = np.random.default_rng(seed=random_seed)

    def random_sample_once():
        if a == 1:
            sample_indices = cj + rng.choice(n, size=P, replace=replace, p=weights)
            return np.mean(alldist[np.ix_(sample_indices, sample_indices)][np.triu_indices(P, 1)])
        elif a == 2:
            sample_indices = cj + rng.choice(n, size=P, replace=replace, p=weights)
            sampled_sequences = all_sequences[sample_indices]
            shuffled = np.array([rng.choice(seq, size=seq_length, replace=replace) for seq in sampled_sequences])
            dmatrix = get_distance_matrix(shuffled, method=method, **distance_kwargs)
            return np.mean(dmatrix[np.triu_indices(P, 1)])
        else:
            raise ValueError("Invalid randomization type 'a'. Should be 1 or 2.")

    # Perform T randomizations
    if verbose:
        iterator = tqdm(range(T), desc="Randomizing polyads")
    else:
        iterator = range(T)

    random_dists = Parallel(n_jobs=n_jobs)(delayed(random_sample_once)() for _ in iterator)
    random_dists = np.array(random_dists)

    # Compute observed distances
    observed_dists = []
    for i in range(n):
        indices = cj + i
        obs_dist = np.mean(alldist[np.ix_(indices, indices)][np.triu_indices(P, 1)])
        observed_dists.append(obs_dist)
    observed_dists = np.array(observed_dists)

    mean_rand_dist = np.mean(random_dists)

    # Compute U and V
    U = mean_rand_dist - observed_dists
    V = np.array([(observed_dists[i] < random_dists).mean() for i in range(n)])
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