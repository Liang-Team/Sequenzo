"""
@Author  : Yuqi Liang 梁彧祺
@File    : ranked_indicators.py
@Time    : 06/02/2026 22:45
@Desc    : Ranked indicators for sequence analysis

    This module implements ranked indicators including:
    - Degradation index (seqidegrad)
    - Badness index (seqibad)
    - Precarity index (seqprecarity)
    - Insecurity index (seqinsecurity)

    Reference: TraMineR R package
    - seqidegrad: R/seqidegrad.R
    - seqibad: R/seqibad.R
    - seqprecarity: R/seqprecarity.R
    - seqprecstart: R/seqprecstart.R (helper function)
    https://github.com/cran/TraMineR/blob/master/R/seqidegrad.R
    https://github.com/cran/TraMineR/blob/master/R/seqibad.R
    https://github.com/cran/TraMineR/blob/master/R/seqprecarity.R
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
import os
from contextlib import redirect_stdout

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.dissimilarity_measures.utils.get_sm_trate_substitution_cost_matrix import get_sm_trate_substitution_cost_matrix
from .integration_index import get_integration_index
from .complexity_index import get_complexity_index
from .simple_characteristics import get_number_of_transitions


def _seqprecstart(seqdata: SequenceData, state_order: List = None, state_equiv: List = None,
                 stprec: np.ndarray = None, with_missing: bool = False) -> tuple:
    """
    Calculate starting state precarity costs.
    
    This is an internal helper function that assigns precarity costs to states
    based on their order or provided costs.
    
    Args:
        seqdata: SequenceData object
        state_order: Ordered list of state indices (1-indexed) from best to worst.
                    If None, uses alphabetical order.
        state_equiv: List of equivalence classes (lists of states that should
                     have the same cost)
        stprec: Predefined precarity costs for each state. If provided,
                state_order is inferred from stprec.
        with_missing: Whether to include missing values
    
    Returns:
        tuple: (stprec array, state_order, state_equiv, state_noncomp)
    """
    alph = seqdata.states.copy()
    alph_size = len(alph)
    
    if state_order is None and stprec is None:
        # Default: use alphabetical order
        state_order = list(range(1, alph_size + 1))
    
    # If stprec provided, infer state_order from it
    if stprec is not None:
        if len(stprec) != alph_size:
            raise ValueError("[!] Length of stprec should equal length of alphabet")
        
        stprec = np.array(stprec)
        # Order states by stprec (ascending: lower cost = better)
        ord_indices = np.argsort(stprec)
        # Only consider positive values
        ord_indices = ord_indices[stprec[ord_indices] >= 0]
        state_order = [i + 1 for i in ord_indices]  # Convert to 1-indexed
        
        # Handle equivalence classes (states with same cost)
        unique_costs = np.unique(stprec[stprec >= 0])
        state_equiv = []
        for cost in unique_costs:
            equiv_states = np.where(stprec == cost)[0] + 1  # Convert to 1-indexed
            if len(equiv_states) > 1:
                state_equiv.append(equiv_states.tolist())
        
        # Normalize negative costs to mean of positive costs
        if np.any(stprec < 0):
            mean_positive = np.mean(stprec[stprec >= 0])
            stprec[stprec < 0] = mean_positive
    
    # Handle non-ranked states (states not in state_order)
    state_noncomp = []
    if state_order is not None:
        unique_order = list(set(state_order))
        if len(unique_order) < alph_size:
            # Some states are not ranked
            all_states = list(range(1, alph_size + 1))
            state_noncomp = [s for s in all_states if s not in unique_order]
            state_order_plus = state_order + state_noncomp
        else:
            state_order_plus = state_order
    else:
        state_order_plus = list(range(1, alph_size + 1))
    
    # Calculate step size for uniform distribution
    if len(state_order_plus) > 1:
        step = 1.0 / (len(state_order_plus) - 1)
    else:
        step = 1.0
    
    # Initialize stprec if not provided
    if stprec is None:
        stprec = np.linspace(0, 1, len(state_order_plus))
        # Assign mean cost to non-ranked states
        if len(state_noncomp) > 0:
            mean_cost = np.mean(stprec[:len(state_order)])
            stprec[len(state_order):] = mean_cost
    
    # Normalize stprec to [0, 1] if provided
    if stprec is not None and (np.min(stprec) < 0 or np.max(stprec) > 1):
        stprec = (stprec - np.min(stprec)) / (np.max(stprec) - np.min(stprec))
    
    # Handle equivalence classes: assign mean cost to all states in class
    if state_equiv is not None:
        for equiv_class in state_equiv:
            equiv_indices = [s - 1 for s in equiv_class]  # Convert to 0-indexed
            mean_cost = np.mean(stprec[equiv_indices])
            stprec[equiv_indices] = mean_cost
    
    return stprec, state_order, state_equiv, state_noncomp


def get_badness_index(seqdata: SequenceData, pow: float = 1.0, with_missing: bool = False,
                     state_order: List = None, state_equiv: List = None,
                     stprec: np.ndarray = None) -> pd.DataFrame:
    """
    Calculate badness index for sequences.
    
    Badness is a weighted sum of integration indices, where weights are
    the precarity costs of states. Higher values indicate "worse" sequences
    (spending more time in states with higher precarity costs).
    
    Formula: sum(stprec[i] * integr[i]) for all states i
    
    Args:
        seqdata: SequenceData object containing sequence data
        pow: Power parameter for integration index calculation
        with_missing: If True, treat missing values as regular states
        state_order: Ordered list of state indices (1-indexed) from best to worst
        state_equiv: List of equivalence classes
        stprec: Predefined precarity costs for each state
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Bad' containing badness values.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqibad.R
        https://github.com/cran/TraMineR/blob/master/R/seqibad.R
    
    Examples:
        >>> badness = get_badness_index(seqdata, pow=1.0)
        >>> print(badness.head())
                 Bad
        ID_1     0.45
        ID_2     0.62
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get precarity costs
    stprec, _, _, _ = _seqprecstart(seqdata, state_order=state_order, state_equiv=state_equiv,
                          stprec=stprec, with_missing=with_missing)
    
    # Get integration indices for all states
    integr = get_integration_index(seqdata, state=None, pow=pow, with_missing=with_missing)
    
    # Calculate badness: weighted sum of integration indices
    # Exclude ID column
    integr_values = integr.iloc[:, 1:].values  # All state columns
    
    # Calculate badness for each sequence
    bad = np.zeros(integr.shape[0])
    for i in range(len(stprec)):
        bad += stprec[i] * integr_values[:, i]
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Bad': bad
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_degradation_index(seqdata: SequenceData, state_order: List = None,
                         state_equiv: List = None, stprec: np.ndarray = None,
                         penalized: str = "BOTH", method: str = "RANK",
                         weight_type: str = "ADD", pow: float = 1.0,
                         border_effect: float = 10.0, spell_integr: bool = True,
                         with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate degradation index for sequences.
    
    Degradation measures the penalty for negative transitions (transitions
    to "worse" states) based on transition probabilities and state ordering.
    
    Args:
        seqdata: SequenceData object containing sequence data
        state_order: Ordered list of state indices (1-indexed) from best to worst
        state_equiv: List of equivalence classes
        stprec: Predefined precarity costs for each state
        penalized: Which transitions to penalize: "NEG", "POS", "BOTH", or "NO"
        method: Method for calculating transition weights:
               "FREQ", "TRATE", "TRATEDSS", "RANK", "FREQ+", "TRATE+", "TRATEDSS+", "RANK+"
        weight_type: Type of weight: "ADD", "INV", or "LOGINV" (only for FREQ/TRATE/TRATEDSS)
        pow: Power parameter for spell integration
        border_effect: Border effect parameter for transition probability adjustment
        spell_integr: If True, use spell integration in weights
        with_missing: If True, treat missing values as regular states
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Degrad' containing degradation values.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqidegrad.R
        https://github.com/cran/TraMineR/blob/master/R/seqidegrad.R
    
    Examples:
        >>> degrad = get_degradation_index(seqdata, method="RANK", penalized="BOTH")
        >>> print(degrad.head())
                 Degrad
        ID_1     0.25
        ID_2     0.42
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Handle logical pow (spell_integr parameter)
    if isinstance(pow, bool):
        spell_integr = pow
        pow = 1.0 if pow else 0.0
    
    # Validate parameters
    method_names = ["FREQ", "TRATE", "TRATEDSS", "RANK", "FREQ+", "TRATE+", "TRATEDSS+", "RANK+", "ONE"]
    if method not in method_names:
        raise ValueError(f"[!] method must be one of {method_names}")
    
    pen_names = ["NEG", "POS", "BOTH", "NO"]
    if penalized not in pen_names:
        raise ValueError(f"[!] penalized must be one of {pen_names}")
    
    if penalized == "NO":
        result = pd.DataFrame({
            'Degrad': np.zeros(seqdata.seqdata.shape[0])
        }, index=seqdata.seqdata.index)
        result = result.reset_index().rename(columns={'index': 'ID'})
        return result
    
    # Handle use.mean.tr flag
    use_mean_tr = False
    if method.endswith("+"):
        use_mean_tr = True
        method = method[:-1]  # Remove the +
    
    # Validate weight_type for transition-based methods
    if method in ["FREQ", "TRATE", "TRATEDSS"]:
        weight_names = ["ADD", "INV", "LOGINV"]
        if weight_type not in weight_names:
            raise ValueError(f"[!] weight_type must be one of {weight_names} when method is {method}")
        if border_effect <= 1:
            raise ValueError("[!] border_effect should be strictly greater than one!")
    
    # Get precarity costs and state ordering
    stprec, state_order, state_equiv, state_noncomp = _seqprecstart(
        seqdata, state_order=state_order, state_equiv=state_equiv,
        stprec=stprec, with_missing=with_missing
    )
    
    alph = seqdata.states.copy()
    alph_size = len(alph)
    
    # Handle non-ranked states: replace them with previous state in sequence
    seq_matrix = seqdata.seqdata.values.copy()
    if state_noncomp:
        # Find positions of non-ranked states
        for i in range(seq_matrix.shape[0]):
            for j in range(1, seq_matrix.shape[1]):
                if seq_matrix[i, j] in state_noncomp:
                    # Replace with previous state
                    seq_matrix[i, j] = seq_matrix[i, j-1]
    
    # Handle equivalence classes: replace all states in class with first state
    if state_equiv:
        for equiv_class in state_equiv:
            first_state = equiv_class[0]
            for state in equiv_class[1:]:
                seq_matrix[seq_matrix == state] = first_state
    
    # Create temporary SequenceData with modified sequences
    temp_seqdata = SequenceData(
        data=pd.DataFrame(seq_matrix, index=seqdata.seqdata.index, columns=seqdata.seqdata.columns),
        time=list(seqdata.seqdata.columns),
        states=seqdata.states,
        id_col=None
    )
    
    # Get DSS and lengths
    dss = seqdss(temp_seqdata)
    dssl = seqlength(dss)  # Returns numpy array
    nbseq = dss.shape[0]
    max_dssl = int(dssl.max()) if len(dssl) > 0 else 1
    
    # Initialize integration matrix
    integr = np.ones((nbseq, max_dssl))
    
    # Calculate spell integration if requested
    if spell_integr:
        # Note: seqdur doesn't support with_missing parameter - missing values are handled at SequenceData level
        Dur = seqdur(temp_seqdata)
        Dur_df = pd.DataFrame(Dur, index=seqdata.seqdata.index)
        
        # Create SPS format sequences for integration calculation
        # Format: "1/dur1 2/dur2 ..."
        for i in range(nbseq):
            dur_row = Dur_df.iloc[i].values
            dur_row = dur_row[~np.isnan(dur_row)]
            if len(dur_row) > 0:
                # Create integration vector for spells
                spell_positions = np.arange(1, len(dur_row) + 1)
                integ_vector = np.power(spell_positions, pow)
                
                # Calculate integration for each spell position
                for j in range(min(len(dur_row), max_dssl)):
                    if j < len(integ_vector):
                        integr[i, j] = integ_vector[j] / np.sum(integ_vector[:j+1])
    
    # Initialize transition weight matrix
    tr = np.ones((alph_size, alph_size))
    np.fill_diagonal(tr, 0)
    signs = np.zeros((alph_size, alph_size))
    
    # Calculate transition weights based on method
    if method in ["FREQ", "TRATE", "TRATEDSS"]:
        # Get transition rates
        if method == "FREQ":
            # Count transitions
            # Note: get_sm_trate_substitution_cost_matrix doesn't support with_missing parameter
            tr_matrix = get_sm_trate_substitution_cost_matrix(
                temp_seqdata,
                weighted=False, count=True
            )
            # Convert to proportions
            tr = tr_matrix / np.sum(tr_matrix)
        elif method == "TRATEDSS":
            # seqdss returns numpy array, need to create SequenceData for get_sm_trate_substitution_cost_matrix
            # Create a temporary SequenceData from DSS array
            # DSS array has variable length rows, pad with -999 (missing value marker)
            max_cols = temp_seqdata.seqdata.shape[1]
            dss_padded = np.full((dss.shape[0], max_cols), -999, dtype=dss.dtype)
            for i in range(dss.shape[0]):
                dss_len = min(dss.shape[1], max_cols)
                dss_padded[i, :dss_len] = dss[i, :dss_len]
            
            dss_df = pd.DataFrame(dss_padded, index=temp_seqdata.seqdata.index,
                                 columns=temp_seqdata.seqdata.columns)
            dss_seqdata = SequenceData(
                data=dss_df,
                time=list(temp_seqdata.seqdata.columns),
                states=temp_seqdata.states,
                id_col=None
            )
            tr = get_sm_trate_substitution_cost_matrix(
                dss_seqdata, weighted=False, count=False
            )
        elif method == "TRATE":
            tr = get_sm_trate_substitution_cost_matrix(
                temp_seqdata, weighted=False, count=False
            )
        
        # Ensure tr is alph_size x alph_size
        if tr.shape[0] != alph_size:
            # Resize if needed (shouldn't happen, but safety check)
            tr_full = np.zeros((alph_size, alph_size))
            min_size = min(tr.shape[0], alph_size)
            tr_full[:min_size, :min_size] = tr[:min_size, :min_size]
            tr = tr_full
        
        np.fill_diagonal(tr, 0)
        
        # Adjust for border effect
        eps = 1e-10
        if np.any(tr > 1 - 0.1 / border_effect):
            tr = tr - tr / border_effect
        
        # Apply weight transformation
        if weight_type == "ADD":
            tr = 1 - tr
        elif weight_type == "INV":
            tr = (1 + eps) / (tr + eps)
        elif weight_type == "LOGINV":
            tr = np.log((1 + eps) / (tr + eps))
        
        # Normalize by diagonal (but diagonal is 0, so normalize by max)
        max_tr = np.max(tr[tr > 0]) if np.any(tr > 0) else 1.0
        if max_tr > 0:
            tr = tr / max_tr
    
    elif method == "RANK":
        # Transition weights based on rank differences
        tr = np.abs(np.subtract.outer(stprec, stprec))
        np.fill_diagonal(tr, 0)
    
    elif method == "ONE":
        # All transitions have equal weight
        tr = np.ones((alph_size, alph_size))
        np.fill_diagonal(tr, 0)
    
    # Set up signs matrix based on penalized parameter
    state_order_plus = state_order + (state_noncomp if state_noncomp else [])
    
    # Create mapping from state_order_plus to alphabet indices
    order_indices = [s - 1 for s in state_order_plus]  # Convert to 0-indexed
    
    # Set signs based on state ordering
    for i in range(alph_size):
        for j in range(alph_size):
            if i in order_indices and j in order_indices:
                i_pos = order_indices.index(i)
                j_pos = order_indices.index(j)
                if penalized == "NEG":
                    if i_pos < j_pos:  # Transition to worse state
                        signs[i, j] = 1
                elif penalized == "POS":
                    if i_pos > j_pos:  # Transition to better state
                        signs[i, j] = -1
                elif penalized == "BOTH":
                    if i_pos < j_pos:  # To worse
                        signs[i, j] = 1
                    elif i_pos > j_pos:  # To better
                        signs[i, j] = -1
    
    # Ignore transitions within equivalence classes
    if state_equiv:
        for equiv_class in state_equiv:
            equiv_indices = [s - 1 for s in equiv_class]
            for idx1 in equiv_indices:
                for idx2 in equiv_indices:
                    signs[idx1, idx2] = 0
                    tr[idx1, idx2] = 0
    
    # Ignore transitions to/from non-ranked states
    if state_noncomp:
        noncomp_indices = [s - 1 for s in state_noncomp]
        for idx in noncomp_indices:
            signs[:, idx] = 0
            signs[idx, :] = 0
            tr[:, idx] = 0
            tr[idx, :] = 0
    
    np.fill_diagonal(tr, 0)
    
    # Calculate degradation for each sequence
    transw = np.zeros(nbseq)
    transpen = np.zeros(nbseq)
    prop_transpen = np.zeros(nbseq)
    
    # seqdss returns a numpy array, not a SequenceData object
    dss_matrix = dss
    
    for i in range(nbseq):
        dssl_i = int(dssl[i]) if i < len(dssl) else 1
        if dssl_i > 1:
            for j in range(1, dssl_i):
                state_from = int(dss_matrix[i, j-1]) - 1  # Convert to 0-indexed
                state_to = int(dss_matrix[i, j]) - 1
                
                if 0 <= state_from < alph_size and 0 <= state_to < alph_size:
                    tr_weight = tr[state_from, state_to]
                    sign_val = signs[state_from, state_to]
                    integr_val = integr[i, j] if j < integr.shape[1] else 1.0
                    
                    transw[i] += tr_weight * integr_val
                    transpen[i] += tr_weight * sign_val * integr_val
    
    # Calculate proportional penalty
    nz = transw > 0
    if spell_integr:
        prop_transpen[nz] = transpen[nz]
    else:
        prop_transpen[nz] = transpen[nz] / transw[nz]
    
    # Apply mean transition weight if requested
    if use_mean_tr:
        mean_transw = transw / dssl
        prop_transpen[nz] = mean_transw[nz] * prop_transpen[nz]
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Degrad': prop_transpen
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_precarity_index(seqdata: SequenceData, correction: np.ndarray = None,
                       state_order: List = None, state_equiv: List = None,
                       stprec: np.ndarray = None, with_missing: bool = False,
                       otto: float = 0.2, a: float = 1.0, b: float = 1.2,
                       method: str = "TRATEDSS", pow: float = 1.0) -> pd.DataFrame:
    """
    Calculate precarity index for sequences.
    
    Precarity combines starting state cost, integration, complexity, and degradation.
    
    Formula: otto * (stprec[start] * integr1) + (1-otto) * (ici^a * (1 + correction)^b)
    
    Args:
        seqdata: SequenceData object containing sequence data
        correction: Pre-computed degradation correction. If None, computed automatically.
        state_order: Ordered list of state indices (1-indexed) from best to worst
        state_equiv: List of equivalence classes
        stprec: Predefined precarity costs for each state
        with_missing: If True, treat missing values as regular states
        otto: Weight parameter for starting state component
        a: Power parameter for complexity component
        b: Power parameter for correction component
        method: Method for degradation calculation
        pow: Power parameter for integration calculation
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Prec' containing precarity values.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqprecarity.R
        https://github.com/cran/TraMineR/blob/master/R/seqprecarity.R
    
    Examples:
        >>> precarity = get_precarity_index(seqdata, otto=0.2, a=1.0, b=1.2)
        >>> print(precarity.head())
                 Prec
        ID_1     0.45
        ID_2     0.62
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get precarity costs
    stprec, state_order, state_equiv, _ = _seqprecstart(
        seqdata, state_order=state_order, state_equiv=state_equiv,
        stprec=stprec, with_missing=with_missing
    )
    
    # Calculate degradation correction if not provided
    if correction is None:
        degrad_result = get_degradation_index(
            seqdata, state_order=state_order,
            state_equiv=state_equiv, stprec=stprec,
            method=method, spell_integr=False,
            with_missing=with_missing
        )
        correction = degrad_result['Degrad'].values
    
    # Get complexity index
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            ici = get_complexity_index(seqdata)
    ici_values = ici.iloc[:, 1].values
    
    # Get starting state for each sequence
    dss = seqdss(seqdata)
    # seqdss returns numpy array, not SequenceData
    dss_matrix = dss
    
    # Get starting state index (0-indexed)
    lalph = dss_matrix[:, 0].astype(int) - 1
    
    # Calculate integration for starting spell
    # For type=1 (precarity), start.integr=FALSE, so integr1 = 1
    integr1 = np.ones(seqdata.seqdata.shape[0])
    
    # Calculate precarity
    prec = otto * (stprec[lalph] * integr1) + (1 - otto) * (ici_values ** a * (1 + correction) ** b)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Prec': prec
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_insecurity_index(seqdata: SequenceData, correction: np.ndarray = None,
                        state_order: List = None, state_equiv: List = None,
                        stprec: np.ndarray = None, with_missing: bool = False,
                        pow: float = 1.0, spow: float = None, bound: bool = False,
                        method: str = "RANK") -> pd.DataFrame:
    """
    Calculate insecurity index for sequences.
    
    Insecurity is similar to precarity but uses different parameters and
    includes spell integration in the starting state component.
    
    Formula: stprec[start] * integr1 + (ici + correction)
    
    Args:
        seqdata: SequenceData object containing sequence data
        correction: Pre-computed degradation correction. If None, computed automatically.
        state_order: Ordered list of state indices (1-indexed) from best to worst
        state_equiv: List of equivalence classes
        stprec: Predefined precarity costs for each state
        with_missing: If True, treat missing values as regular states
        pow: Power parameter for integration calculation
        spow: Power parameter for starting spell integration (defaults to pow)
        bound: If True, bound the result between min and max state costs
        method: Method for degradation calculation
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Insec' containing insecurity values.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqprecarity.R (type=2)
        https://github.com/cran/TraMineR/blob/master/R/seqprecarity.R
    
    Examples:
        >>> insecurity = get_insecurity_index(seqdata, pow=1.0, bound=False)
        >>> print(insecurity.head())
                 Insec
        ID_1     0.55
        ID_2     0.72
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    if spow is None:
        spow = pow
    
    # Get precarity costs
    stprec, state_order, state_equiv, _ = _seqprecstart(
        seqdata, state_order=state_order, state_equiv=state_equiv,
        stprec=stprec, with_missing=with_missing
    )
    
    # Calculate degradation correction if not provided
    if correction is None:
        degrad_result = get_degradation_index(
            seqdata, state_order=state_order,
            state_equiv=state_equiv, stprec=stprec,
            method=method, spell_integr=True,
            pow=pow, with_missing=with_missing
        )
        correction = degrad_result['Degrad'].values
    
    # Get complexity index
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            ici = get_complexity_index(seqdata)
    ici_values = ici.iloc[:, 1].values
    
    # Get starting state and DSS
    dss = seqdss(seqdata)  # Returns numpy array
    dss_matrix = dss
    dssl = seqlength(dss)
    
    # Get starting state index (0-indexed)
    lalph = dss_matrix[:, 0].astype(int) - 1
    
    # Calculate integration for starting spell (spell.integr=TRUE for type=2)
    # Note: seqdur doesn't support with_missing parameter - missing values are handled at SequenceData level
    Dur = seqdur(seqdata)
    Dur_df = pd.DataFrame(Dur, index=seqdata.seqdata.index)
    
    integr1 = np.ones(seqdata.seqdata.shape[0])
    for i in range(seqdata.seqdata.shape[0]):
        dur_row = Dur_df.iloc[i].values
        dur_row = dur_row[~np.isnan(dur_row)]
        if len(dur_row) > 0 and dur_row[0] > 0:
            # Calculate integration for first spell
            spell_positions = np.arange(1, int(dur_row[0]) + 1)
            integ_vector = np.power(spell_positions, spow)
            sum_integ = np.sum(integ_vector)
            if sum_integ > 0:
                # Integration weight for first position
                integr1[i] = integ_vector[0] / sum_integ
    
    # Calculate insecurity
    insec = stprec[lalph] * integr1 + (ici_values + correction)
    
    # Apply bounds if requested
    if bound:
        alph = seqdata.states.copy()
        
        # Get min and max state costs for each sequence's DSS
        min_costs = np.full(seqdata.seqdata.shape[0], np.min(stprec))
        max_costs = np.full(seqdata.seqdata.shape[0], np.max(stprec))
        
        for i in range(seqdata.seqdata.shape[0]):
            dssl_i = int(dssl[i]) if i < len(dssl) else 1
            dss_states = dss_matrix[i, :dssl_i].astype(int) - 1
            if len(dss_states) > 0:
                valid_states = dss_states[dss_states >= 0]
                if len(valid_states) > 0:
                    min_costs[i] = np.min(stprec[valid_states])
                    max_costs[i] = np.max(stprec[valid_states])
        
        # Bound the result
        insec = np.maximum(insec, min_costs)
        insec = np.minimum(insec, max_costs)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Insec': insec
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result
