"""
@Author  : Yapeng Wei 卫亚鹏
@File    : estimate_mnhmm.py
@Time    : 2026-05-25 02:04
@Desc    : Estimate Mixture Non-homogeneous Hidden Markov Models (MNHMMs) for Sequenzo

Estimate mixture non-homogeneous HMMs.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData

from .build_mnhmm import build_mnhmm
from .formulas import Formula
from .mnhmm import EmissionProbabilities, StateNames


def estimate_mnhmm(
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
    initial_probs: Optional[Sequence[np.ndarray]] = None,
    transition_probs: Optional[Sequence[np.ndarray]] = None,
    emission_probs: Optional[EmissionProbabilities] = None,
    cluster_probs: Optional[np.ndarray] = None,
    eta_pi_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_A_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_B_reduced: Optional[Sequence[np.ndarray]] = None,
    eta_omega_reduced: Optional[np.ndarray] = None,
    cluster_names: Optional[Sequence[str]] = None,
    state_names: Optional[StateNames] = None,
    random_state: Optional[int] = None,
    n_iter: int = 100,
    tol: float = 1e-2,
    lambda_penalty: float = 0.0,
    verbose: bool = False,
    probability_parameters_as_starts: bool = False,
    compress: bool = False,
):
    """
    Estimate an MNHMM.

    Intercept-only models with unfixed component probabilities use weighted
    Baum-Welch EM. Single-channel models with covariates in the initial,
    transition, emission, or cluster probabilities use direct L-BFGS
    observed-likelihood optimization with optional L2 ``lambda_penalty``.
    Multichannel MNHMMs support fixed component inference, fixed-component
    cluster-covariate optimization, non-covariate component EM, and direct
    component-covariate likelihood fits. ``compress`` is an opt-in
    acceleration for repeated complete observation patterns in fixed-probability
    EM and fixed-component cluster-covariate fits.

    By default, supplied probability arrays are treated as fixed probabilities,
    matching ``build_mnhmm`` for direct covariate fits. Set
    ``probability_parameters_as_starts=True`` to mirror R seqHMM fitting
    semantics, where supplied probabilities initialize covariate-model eta
    coefficients instead of freezing those families.
    """
    model = build_mnhmm(
        observations=observations,
        n_states=n_states,
        n_clusters=n_clusters,
        X=X,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        X_cluster=X_cluster,
        emission_formula=emission_formula,
        initial_formula=initial_formula,
        transition_formula=transition_formula,
        cluster_formula=cluster_formula,
        data=data,
        id_var=id_var,
        time_var=time_var,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=cluster_probs,
        eta_pi_reduced=eta_pi_reduced,
        eta_A_reduced=eta_A_reduced,
        eta_B_reduced=eta_B_reduced,
        eta_omega_reduced=eta_omega_reduced,
        cluster_names=cluster_names,
        state_names=state_names,
        random_state=random_state,
    )
    direct_needed = (not model._supports_fixed_probability_em()) or model.has_cluster_covariates()
    if direct_needed and probability_parameters_as_starts:
        model.use_probability_parameters_as_covariate_starts()
    if direct_needed and cluster_probs is None:
        model.enable_cluster_covariate_estimation(random_state=random_state)
    return model.fit(
        n_iter=n_iter,
        tol=tol,
        verbose=verbose,
        lambda_penalty=lambda_penalty,
        compress=compress,
    )
