"""Helpers for comparing HMM parameters up to hidden-state label switching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


ParameterDict = Dict[str, object]


def _emission_rows(emission) -> np.ndarray:
    if isinstance(emission, list):
        return np.hstack([np.asarray(channel, dtype=float) for channel in emission])
    return np.asarray(emission, dtype=float)


def _state_cost_matrix(candidate: ParameterDict, reference: ParameterDict) -> np.ndarray:
    candidate_initial = np.asarray(candidate["initial"], dtype=float)
    reference_initial = np.asarray(reference["initial"], dtype=float)
    candidate_emission = _emission_rows(candidate["emission"])
    reference_emission = _emission_rows(reference["emission"])

    n_states = len(reference_initial)
    cost = np.zeros((n_states, n_states), dtype=float)
    for cand_state in range(n_states):
        for ref_state in range(n_states):
            cost[cand_state, ref_state] = (
                abs(candidate_initial[cand_state] - reference_initial[ref_state])
                + np.linalg.norm(
                    candidate_emission[cand_state] - reference_emission[ref_state],
                    ord=1,
                )
            )
    return cost


def _apply_permutation(params: ParameterDict, permutation: List[int]) -> ParameterDict:
    permutation = np.asarray(permutation, dtype=int)
    emission = params["emission"]
    if isinstance(emission, list):
        aligned_emission = [np.asarray(channel)[permutation] for channel in emission]
    else:
        aligned_emission = np.asarray(emission)[permutation]

    return {
        "initial": np.asarray(params["initial"])[permutation],
        "transition": np.asarray(params["transition"])[np.ix_(permutation, permutation)],
        "emission": aligned_emission,
    }


def align_hmm_state_labels(
    candidate: ParameterDict,
    reference: ParameterDict,
) -> Tuple[ParameterDict, List[int]]:
    """
    Reorder candidate states to match reference states via Hungarian assignment.

    Returns the aligned candidate parameters and the candidate-state order used
    to produce the reference-state order.
    """
    cost = _state_cost_matrix(candidate, reference)
    candidate_rows, reference_cols = linear_sum_assignment(cost)

    permutation = [None] * len(reference_cols)
    for candidate_state, reference_state in zip(candidate_rows, reference_cols):
        permutation[reference_state] = int(candidate_state)

    return _apply_permutation(candidate, permutation), permutation
