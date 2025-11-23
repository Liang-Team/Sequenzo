"""
@Author  : Yuqi Liang 梁彧祺
@File    : predict.py
@Time    : 2025-11-13 17:05
@Desc    : Prediction and inference functions for HMM models

This module provides functions for predicting hidden states and computing
posterior probabilities, similar to seqHMM's predict() and posterior_probs()
functions in R.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from sequenzo.define_sequence_data import SequenceData
from .hmm import HMM
from .utils import sequence_data_to_hmmlearn_format


def predict(
    model: HMM,
    newdata: Optional[SequenceData] = None
) -> np.ndarray:
    """
    Predict the most likely hidden state sequence using Viterbi algorithm.
    
    This function finds the most likely sequence of hidden states given the
    observed sequence, using the Viterbi algorithm (dynamic programming).
    
    It is similar to seqHMM's predict() function in R.
    
    Args:
        model: Fitted HMM model object
        newdata: Optional SequenceData to predict. If None, uses the data
                 the model was fitted on.
        
    Returns:
        numpy array: Predicted hidden states for each time point in each sequence.
                    The array is flattened (all sequences concatenated).
                    
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_hmm, fit_model, predict
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> 
        >>> # Predict hidden states
        >>> predicted_states = predict(hmm)
        >>> print(f"Predicted {len(predicted_states)} hidden states")
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before prediction. Use fit_model() first.")
    
    return model.predict(newdata)


def posterior_probs(
    model: HMM,
    newdata: Optional[SequenceData] = None
) -> pd.DataFrame:
    """
    Compute posterior probabilities of hidden states.
    
    This function computes the probability of each hidden state at each time point,
    given the observed sequence. It uses the forward-backward algorithm.
    
    It is similar to seqHMM's posterior_probs() function in R.
    
    Args:
        model: Fitted HMM model object
        newdata: Optional SequenceData. If None, uses the data the model was fitted on.
        
    Returns:
        pandas DataFrame: Posterior probabilities with columns:
            - id: Sequence identifier (index in the original data)
            - time: Time point within the sequence
            - state: Hidden state index
            - probability: Posterior probability of being in this state at this time
            
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_hmm, fit_model, posterior_probs
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> 
        >>> # Get posterior probabilities
        >>> posteriors = posterior_probs(hmm)
        >>> print(posteriors.head())
        >>> 
        >>> # Find most probable state at each time point
        >>> most_probable = posteriors.groupby(['id', 'time'])['probability'].idxmax()
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before computing posterior probabilities. Use fit_model() first.")
    
    # Get sequences to use
    sequences = newdata if newdata is not None else model.observations
    
    # Get posterior probabilities from model
    proba = model.predict_proba(sequences)
    
    # Get sequence information
    X, lengths = sequence_data_to_hmmlearn_format(sequences)
    
    # Create DataFrame with results
    rows = []
    seq_idx = 0
    time_idx = 0
    
    for seq_id in range(len(lengths)):
        seq_length = lengths[seq_id]
        for t in range(seq_length):
            for state_idx in range(model.n_states):
                rows.append({
                    'id': seq_id,
                    'time': t + 1,  # 1-indexed for consistency with R
                    'state': state_idx,
                    'probability': proba[time_idx, state_idx]
                })
            time_idx += 1
        seq_idx += 1
    
    df = pd.DataFrame(rows)
    
    return df
