"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2025-11-13 19:27
@Desc    : Hidden Markov Models for sequence analysis in Sequenzo

This module provides Hidden Markov Model (HMM) functionality for sequence analysis,
inspired by the seqHMM R package but implemented natively in Python using hmmlearn.

Main features:
- Basic HMM: Standard hidden Markov models for sequence data
- Model building: Create HMM models from SequenceData
- Parameter estimation: Fit models using EM algorithm
- Prediction: Predict hidden states and compute posterior probabilities
- Visualization: Plot HMM models and results
"""

from .hmm import HMM
from .build_hmm import build_hmm
from .fit_model import fit_model
from .predict import predict, posterior_probs
from .visualization import plot_hmm

# Mixture HMM
from .mhmm import MHMM
from .build_mhmm import build_mhmm
from .fit_mhmm import fit_mhmm
from .predict_mhmm import predict_mhmm, posterior_probs_mhmm
from .visualization import plot_mhmm

# Non-homogeneous HMM
from .nhmm import NHMM
from .build_nhmm import build_nhmm
from .fit_nhmm import fit_nhmm

# Model comparison and simulation
from .model_comparison import aic, bic, compare_models, compute_n_parameters, compute_n_observations
from .simulate import simulate_hmm, simulate_mhmm, simulate_nhmm
from .bootstrap import bootstrap_model

# Forward-backward for NHMM
from .forward_backward_nhmm import forward_backward_nhmm, log_likelihood_nhmm

# Gradients for NHMM
from .gradients_nhmm import compute_gradient_nhmm

# Formulas for NHMM and MHMM simulation
from .formulas import Formula, create_model_matrix, create_model_matrix_time_constant

# Advanced optimization
from .advanced_optimization import fit_model_advanced

__all__ = [
    # Basic HMM
    'HMM',
    'build_hmm',
    'fit_model',
    'predict',
    'posterior_probs',
    'plot_hmm',
    # Mixture HMM
    'MHMM',
    'build_mhmm',
    'fit_mhmm',
    'predict_mhmm',
    'posterior_probs_mhmm',
    'plot_mhmm',
    # Non-homogeneous HMM
    'NHMM',
    'build_nhmm',
    'fit_nhmm',
    # Model comparison
    'aic',
    'bic',
    'compare_models',
    'compute_n_parameters',
    'compute_n_observations',
    # Simulation
    'simulate_hmm',
    'simulate_mhmm',
    'simulate_nhmm',
    # Bootstrap
    'bootstrap_model',
    # Forward-backward for NHMM
    'forward_backward_nhmm',
    'log_likelihood_nhmm',
    # Gradients for NHMM
    'compute_gradient_nhmm',
    # Formulas for NHMM and MHMM simulation
    'Formula',
    'create_model_matrix',
    'create_model_matrix_time_constant',
    # Advanced optimization
    'fit_model_advanced',
]
