"""
@Author  : Yapeng Wei 卫亚鹏
@File    : build_mnhmm.py
@Time    : 2026-05-25 02:04
@Desc    : Build Mixture Non-homogeneous Hidden Markov Models (MNHMMs) for Sequenzo

Build mixture non-homogeneous hidden Markov models.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData

from .formulas import Formula, create_model_matrix, create_model_matrix_time_constant
from .mnhmm import EmissionProbabilities, MNHMM, StateNames


def _rhs_formula(formula: Optional[Union[str, Formula]]) -> Optional[Union[str, Formula]]:
    if formula is None or isinstance(formula, Formula):
        return formula
    text = str(formula).strip()
    if "~" not in text:
        return text
    rhs = text.split("~", 1)[1].strip() or "1"
    return f"~ {rhs}"


def _matrix_from_formula(
    formula: Optional[Union[str, Formula]],
    data: Optional[pd.DataFrame],
    id_var: Optional[str],
    time_var: Optional[str],
    observations: SequenceData,
) -> Optional[np.ndarray]:
    if formula is None:
        return None
    if data is None or id_var is None or time_var is None:
        raise ValueError("Formula-based MNHMM construction requires data, id_var, and time_var")

    n_sequences = len(observations.sequences)
    n_timepoints = max(len(seq) for seq in observations.sequences)
    return create_model_matrix(
        _rhs_formula(formula),
        data,
        id_var,
        time_var,
        n_sequences,
        n_timepoints,
        id_values=observations.ids,
        time_values=observations.time,
    )


def _cluster_matrix_from_formula(
    formula: Optional[Union[str, Formula]],
    data: Optional[pd.DataFrame],
    id_var: Optional[str],
    time_var: Optional[str],
    observations: SequenceData,
) -> Optional[np.ndarray]:
    if formula is None:
        return None
    if data is None or id_var is None:
        raise ValueError("cluster_formula requires data and id_var")

    n_sequences = len(observations.sequences)
    n_timepoints = max(len(seq) for seq in observations.sequences)
    if time_var is not None and time_var in data.columns and data.duplicated([id_var]).any():
        X_time = create_model_matrix(
            _rhs_formula(formula),
            data,
            id_var,
            time_var,
            n_sequences,
            n_timepoints,
            id_values=observations.ids,
            time_values=observations.time,
        )
        if not np.allclose(X_time, X_time[:, :1, :]):
            raise ValueError("cluster_formula covariates must be time-constant")
        return X_time[:, 0, :]

    if id_var not in data.columns:
        raise ValueError(f"data must contain id_var={id_var!r}")
    if data.duplicated([id_var]).any():
        raise ValueError("cluster_formula data must contain one row per sequence id")

    ids = list(observations.ids)
    data_ids = set(data[id_var])
    expected_ids = set(ids)
    if data_ids != expected_ids:
        raise ValueError(
            "cluster_formula data must contain exactly the SequenceData ids"
        )

    ordered = data.set_index(id_var, drop=False).loc[ids].reset_index(drop=True)
    return create_model_matrix_time_constant(_rhs_formula(formula), ordered, n_sequences)


def build_mnhmm(
    observations: Union[SequenceData, Sequence[SequenceData]],
    n_states: Union[int, Sequence[int]],
    n_clusters: int,
    X: Optional[np.ndarray] = None,
    X_pi: Optional[np.ndarray] = None,
    X_A: Optional[np.ndarray] = None,
    X_B: Optional[np.ndarray] = None,
    X_cluster: Optional[np.ndarray] = None,
    emission_formula: Optional[Union[str, Formula]] = None,
    initial_formula: Optional[Union[str, Formula]] = None,
    transition_formula: Optional[Union[str, Formula]] = None,
    cluster_formula: Optional[Union[str, Formula]] = None,
    data: Optional[pd.DataFrame] = None,
    id_var: Optional[str] = None,
    time_var: Optional[str] = None,
    eta_pi: Optional[Sequence[np.ndarray]] = None,
    eta_A: Optional[Sequence[np.ndarray]] = None,
    eta_B: Optional[Sequence[np.ndarray]] = None,
    eta_omega: Optional[np.ndarray] = None,
    eta_pi_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_A_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_B_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_omega_reduced: Optional[np.ndarray] = None,
    initial_probs: Optional[Sequence[np.ndarray]] = None,
    transition_probs: Optional[Sequence[np.ndarray]] = None,
    emission_probs: Optional[EmissionProbabilities] = None,
    cluster_probs: Optional[np.ndarray] = None,
    cluster_names: Optional[Sequence[str]] = None,
    state_names: Optional[StateNames] = None,
    random_state: Optional[int] = None,
) -> MNHMM:
    """
    Build an MNHMM object without estimating parameters.

    The builder accepts direct covariate matrices or formulas, plus fixed
    component probabilities when parameters are already known. For
    multichannel observations, pass emission probabilities as
    ``emission_probs[cluster][channel]``.
    """
    if int(n_clusters) < 2:
        raise ValueError("n_clusters must be at least 2")

    observations_input = list(observations) if isinstance(observations, tuple) else observations
    primary_observations = (
        observations_input[0]
        if isinstance(observations_input, list)
        else observations_input
    )
    if X_pi is None:
        X_pi = _matrix_from_formula(initial_formula, data, id_var, time_var, primary_observations)
    if X_A is None:
        X_A = _matrix_from_formula(transition_formula, data, id_var, time_var, primary_observations)
    if X_B is None:
        X_B = _matrix_from_formula(emission_formula, data, id_var, time_var, primary_observations)
    if X_cluster is None:
        X_cluster_formula = _cluster_matrix_from_formula(
            cluster_formula, data, id_var, time_var, primary_observations
        )
        if X_cluster_formula is not None:
            X_cluster = X_cluster_formula

    return MNHMM(
        observations=observations_input,
        n_clusters=n_clusters,
        n_states=n_states,
        X=X,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        X_cluster=X_cluster,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
        eta_omega=eta_omega,
        eta_pi_reduced=eta_pi_reduced,
        eta_A_reduced=eta_A_reduced,
        eta_B_reduced=eta_B_reduced,
        eta_omega_reduced=eta_omega_reduced,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=cluster_probs,
        cluster_names=cluster_names,
        state_names=state_names,
        random_state=random_state,
    )
