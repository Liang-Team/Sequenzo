"""
@Author  : Yuqi Liang 梁彧祺
@File    : predict_mhmm.py
@Time    : 2025-11-22 11:03
@Desc    : Prediction functions for Mixture HMM models

This module provides functions for predicting cluster assignments and computing
posterior probabilities for Mixture HMM models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sequenzo.define_sequence_data import SequenceData
from .mhmm import MHMM


def predict_mhmm(
    model: MHMM,
    newdata: Optional[Union[SequenceData, List[SequenceData]]] = None,
    compress: bool = False,
) -> np.ndarray:
    """
    Predict the most likely cluster for each sequence.
    
    This function finds the most likely cluster assignment for each sequence
    based on the fitted Mixture HMM model.
    
    Args:
        model: Fitted MHMM model object
        newdata: Optional SequenceData to predict. If None, uses the data
                 the model was fitted on.
        compress: Whether to reuse likelihoods for repeated sequences
        
    Returns:
        numpy array: Predicted cluster index for each sequence
                    
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_mhmm, fit_mhmm, predict_mhmm
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> mhmm = build_mhmm(seq, n_clusters=3, n_states=4, random_state=42)
        >>> mhmm = fit_mhmm(mhmm)
        >>> 
        >>> # Predict clusters
        >>> predicted_clusters = predict_mhmm(mhmm)
        >>> print(f"Predicted clusters: {predicted_clusters}")
    """
    if model.log_likelihood is None and not model.has_complete_parameters:
        raise ValueError(
            "Model must be fitted before prediction, or all cluster HMMs "
            "must have complete fixed parameters."
        )

    responsibilities = model.compute_responsibilities(newdata, compress=compress)
    return np.argmax(responsibilities, axis=1)


def posterior_probs_mhmm(
    model: MHMM,
    newdata: Optional[Union[SequenceData, List[SequenceData]]] = None,
    compress: bool = False,
) -> pd.DataFrame:
    """
    Compute posterior probabilities of cluster membership.
    
    This function computes the probability that each sequence belongs to each
    cluster, given the observed sequence.
    
    Args:
        model: Fitted MHMM model object
        newdata: Optional SequenceData. If None, uses the data the model was fitted on.
        
    Returns:
        pandas DataFrame: Posterior probabilities with columns:
            - id: Sequence identifier (index in the original data)
            - cluster: Cluster name
            - probability: Posterior probability of belonging to this cluster
            
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_mhmm, fit_mhmm, posterior_probs_mhmm
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> mhmm = build_mhmm(seq, n_clusters=3, n_states=4, random_state=42)
        >>> mhmm = fit_mhmm(mhmm)
        >>> 
        >>> # Get posterior probabilities
        >>> posteriors = posterior_probs_mhmm(mhmm)
        >>> print(posteriors.head())
    """
    if model.log_likelihood is None and not model.has_complete_parameters:
        raise ValueError(
            "Model must be fitted before computing posterior probabilities, "
            "or all cluster HMMs must have complete fixed parameters."
        )

    responsibilities = model.compute_responsibilities(newdata, compress=compress)
    sequences = newdata if newdata is not None else model._default_sequences()
    primary = sequences[0] if isinstance(sequences, list) else sequences
    ids = np.asarray(getattr(primary, "ids", np.arange(responsibilities.shape[0])))
    clusters = np.asarray(model.cluster_names)

    return pd.DataFrame({
        'id': np.repeat(ids, model.n_clusters),
        'cluster': np.tile(clusters, len(ids)),
        'probability': responsibilities.reshape(-1),
    })
