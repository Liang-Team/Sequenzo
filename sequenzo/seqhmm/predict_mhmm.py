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
from typing import Optional
from sequenzo.define_sequence_data import SequenceData
from .mhmm import MHMM
from .utils import sequence_data_to_hmmlearn_format


def predict_mhmm(
    model: MHMM,
    newdata: Optional[SequenceData] = None
) -> np.ndarray:
    """
    Predict the most likely cluster for each sequence.
    
    This function finds the most likely cluster assignment for each sequence
    based on the fitted Mixture HMM model.
    
    Args:
        model: Fitted MHMM model object
        newdata: Optional SequenceData to predict. If None, uses the data
                 the model was fitted on.
        
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
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before prediction. Use fit_mhmm() first.")
    
    return model.predict_cluster(newdata)


def posterior_probs_mhmm(
    model: MHMM,
    newdata: Optional[SequenceData] = None
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
            - cluster: Cluster index
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
    if model.log_likelihood is None:
        raise ValueError(
            "Model must be fitted before computing posterior probabilities. Use fit_mhmm() first."
        )
    
    # Get sequences to use
    sequences = newdata if newdata is not None else model.observations
    
    # Get responsibilities (posterior cluster probabilities)
    if newdata is None:
        # Use stored responsibilities from fitting
        responsibilities = model.responsibilities
    else:
        # Compute responsibilities for new sequences
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        n_sequences = len(lengths)
        
        log_likelihoods = np.zeros((n_sequences, model.n_clusters))
        
        for k in range(model.n_clusters):
            for seq_idx in range(n_sequences):
                start_idx = lengths[:seq_idx].sum()
                end_idx = start_idx + lengths[seq_idx]
                seq_X = X[start_idx:end_idx]
                seq_lengths = np.array([lengths[seq_idx]])
                
                log_likelihoods[seq_idx, k] = model.clusters[k]._hmm_model.score(seq_X, seq_lengths)
        
        log_probs = np.log(model.cluster_probs + 1e-10)
        log_likelihoods += log_probs[np.newaxis, :]
        
        max_log_lik = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_log_lik = np.exp(log_likelihoods - max_log_lik)
        responsibilities = exp_log_lik / np.sum(exp_log_lik, axis=1, keepdims=True)
    
    # Create DataFrame
    rows = []
    for seq_id in range(len(responsibilities)):
        for cluster_idx in range(model.n_clusters):
            rows.append({
                'id': seq_id,
                'cluster': cluster_idx,
                'probability': responsibilities[seq_id, cluster_idx]
            })
    
    df = pd.DataFrame(rows)
    
    return df
