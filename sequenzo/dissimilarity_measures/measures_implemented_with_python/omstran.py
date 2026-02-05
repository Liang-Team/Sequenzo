"""
@Author  : Yuqi Liang 梁彧祺
@File    : omstran.py
@Time    : 2026/02/05 8:15
@Desc    : OMstran (Optimal Matching with transitions) implementation
           Transforms sequences to transition states and computes OM distance
           References:
           OMstran.R from TraMineR package. https://github.com/cran/TraMineR/blob/master/R/seqdist-OMstran.R
"""
import numpy as np
import pandas as pd
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.get_sm_trate_substitution_cost_matrix import get_sm_trate_substitution_cost_matrix
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength


def create_transition_sequences(seqdata, previous=False, add_column=False):
    """
    Create transition states from sequences.
    
    Args:
        seqdata: SequenceData object
        previous: If True, include previous state in transition (creates 3-state transitions)
        add_column: If True, add an extra column
    
    Returns:
        newseqdata: DataFrame with transition states
        void_code: Code for void/missing values
    """
    seqdata_matrix = seqdata.values.copy()
    nseqs, maxcol = seqdata_matrix.shape
    void_code = len(seqdata.states) + 1  # Missing value code
    
    # Separator for transition states
    sep = "@@@@TraMineRSep@@@@"
    
    # Initialize new sequence data matrix
    newseqdata_list = []
    
    def minmax(i):
        """Helper to ensure index is within bounds"""
        return max(0, min(i, maxcol - 1))
    
    for i in range(maxcol):
        # Get from state (current position)
        from_state = seqdata_matrix[:, minmax(i)]
        
        # Get to state (next position)
        to_state = seqdata_matrix[:, minmax(i + 1)]
        
        # Handle void states at end of sequences
        if add_column:
            # When to state is void, set it as the previous state
            to_state = np.where(to_state == void_code, from_state, to_state)
        
        # Create transition strings
        transitions = []
        for f, t in zip(from_state, to_state):
            if f == void_code or t == void_code:
                transitions.append(np.nan)
            else:
                transitions.append(f"{int(f)}{sep}{int(t)}")
        transitions = np.array(transitions)
        
        # Set transitions as NA when appropriate
        if add_column:
            # Set transition as NA when from state is void
            transitions[from_state == void_code] = np.nan
        else:
            # Set transition as NA when to state is void
            transitions[to_state == void_code] = np.nan
        
        # Handle previous state if needed
        if previous:
            prev_state = seqdata_matrix[:, minmax(i - 1)]
            transitions_na = pd.isna(transitions)
            new_transitions = []
            for p, t, is_na in zip(prev_state, transitions, transitions_na):
                if is_na or p == void_code:
                    new_transitions.append(np.nan)
                else:
                    new_transitions.append(f"{int(p)}{sep}{t}")
            transitions = np.array(new_transitions)
        
        newseqdata_list.append(transitions)
    
    # Convert to DataFrame
    newseqdata = pd.DataFrame(np.array(newseqdata_list).T)
    
    # Remove columns if needed
    if not add_column:
        newseqdata = newseqdata.iloc[:, :-1]
        if previous:
            newseqdata = newseqdata.iloc[:, 1:]
    
    return newseqdata, void_code


def build_omstran_substitution_matrix(seqdata, newseqdata, sm, indel, transindel, otto, previous, void_code):
    """
    Build substitution matrix for transition states.
    
    Args:
        seqdata: Original SequenceData object
        newseqdata: DataFrame with transition states
        sm: Original substitution cost matrix
        indel: Original indel costs
        transindel: Method for transition indel ("constant", "prob", "subcost")
        otto: Weight for substitution vs transition indel
        previous: Whether previous state is included
        void_code: Code for void/missing values
    
    Returns:
        newsm: New substitution matrix for transition states
        newindels: New indel costs for transition states
        newalph: New alphabet (transition states)
    """
    # Get alphabet from new sequences
    newalph = []
    for col in newseqdata.columns:
        unique_vals = newseqdata[col].dropna().unique()
        newalph.extend([str(v) for v in unique_vals if pd.notna(v)])
    newalph = sorted(set(newalph))
    alphabet_size = len(newalph)
    
    print(f"[>] Creating {alphabet_size} distinct transition states")
    
    # Normalize sm and indel
    sm_max = np.max(sm)
    sm_normalized = sm / sm_max
    
    # Handle indel vector
    if isinstance(indel, (int, float)):
        indel_vector = np.repeat(indel, len(seqdata.states))
    else:
        indel_vector = np.array(indel)
    
    indel_max = np.max(indel_vector)
    indelrate = indel_max / sm_max
    
    # Compute transition weight
    transweight = (1 - otto) * indelrate
    if previous:
        transweight = transweight / 2
    
    # Normalize indel vector
    indel_normalized = indelrate * indel_vector / indel_max
    
    # Get transition rates if needed
    tr = None
    if transindel == "prob":
        tr = get_sm_trate_substitution_cost_matrix(seqdata, time_varying=False, weighted=True)
        # Convert to numpy array if DataFrame
        if isinstance(tr, pd.DataFrame):
            tr = tr.values
    
    # Build state code-to-index mapping
    # SequenceData uses 1-indexed codes: state_mapping maps states to codes 1, 2, 3, ...
    # We need to map codes to 0-indexed matrix indices
    code_to_idx = {}
    for i, state in enumerate(seqdata.states):
        code = seqdata.state_mapping[state]  # Get 1-indexed code
        code_to_idx[code] = i  # Map to 0-indexed matrix position
    
    # Initialize new substitution matrix and indels
    newsm = np.zeros((alphabet_size, alphabet_size))
    indels = np.zeros(alphabet_size)
    
    sep = "@@@@TraMineRSep@@@@"
    stateindel = indel_normalized * otto
    
    # Compute indel costs for each transition state
    for i, trans_state in enumerate(newalph):
        states_parts = trans_state.split(sep)
        
        if previous:
            # Three states: prev_state, from_state, to_state
            if len(states_parts) == 3:
                prev_s = states_parts[0]
                from_s = states_parts[1]
                to_s = states_parts[2]
                
                # Get state indices (codes are 1-indexed)
                try:
                    prev_code = int(prev_s)
                    from_code = int(from_s)
                    to_code = int(to_s)
                    prev_idx = code_to_idx.get(prev_code, None)
                    from_idx = code_to_idx.get(from_code, None)
                    to_idx = code_to_idx.get(to_code, None)
                except (ValueError, KeyError):
                    continue
                
                if from_idx is None:
                    continue
                
                # Compute raw transition indel
                rawtransindel = 0
                if transindel == "constant":
                    rawtransindel = (prev_code != from_code) + (from_code != to_code)
                elif transindel == "prob" and tr is not None:
                    if prev_idx is not None and to_idx is not None:
                        # tr matrix uses state codes directly as indices (1-indexed)
                        # Matrix shape is (nstates+1, nstates+1) where index 0 is null
                        rawtransindel = 2 - tr[prev_code, from_code] - tr[from_code, to_code]
                elif transindel == "subcost":
                    if prev_idx is not None and to_idx is not None:
                        rawtransindel = sm_normalized[prev_idx, from_idx] + sm_normalized[from_idx, to_idx]
                
                indels[i] = stateindel[from_idx] + transweight * rawtransindel
        else:
            # Two states: from_state, to_state
            if len(states_parts) == 2:
                from_s = states_parts[0]
                to_s = states_parts[1]
                
                try:
                    from_code = int(from_s)
                    to_code = int(to_s)
                    from_idx = code_to_idx.get(from_code, None)
                    to_idx = code_to_idx.get(to_code, None)
                except (ValueError, KeyError):
                    continue
                
                if from_idx is None:
                    continue
                
                indels[i] = stateindel[from_idx]
                
                if from_code != to_code:
                    # Compute raw transition indel
                    rawtransindel = 0
                    if transindel == "constant":
                        rawtransindel = 1
                    elif transindel == "prob" and tr is not None and to_idx is not None:
                        # tr matrix uses state codes directly as indices (1-indexed)
                        rawtransindel = 1 - tr[from_code, to_code]
                    elif transindel == "subcost" and to_idx is not None:
                        rawtransindel = sm_normalized[from_idx, to_idx]
                    
                    indels[i] += transweight * rawtransindel
    
    # Build substitution matrix
    compare_state_idx = 2 if previous else 1  # Index of state to compare
    
    for i in range(alphabet_size - 1):
        states_i = newalph[i].split(sep)
        if len(states_i) < compare_state_idx + 1:
            continue
        
        try:
            state_i_code = int(states_i[compare_state_idx])
            state_i_idx = code_to_idx.get(state_i_code, None)
        except (ValueError, KeyError):
            continue
        
        if state_i_idx is None:
            continue
        
        for j in range(i + 1, alphabet_size):
            states_j = newalph[j].split(sep)
            if len(states_j) < compare_state_idx + 1:
                continue
            
            try:
                state_j_code = int(states_j[compare_state_idx])
                state_j_idx = code_to_idx.get(state_j_code, None)
            except (ValueError, KeyError):
                continue
            
            if state_j_idx is None:
                continue
            
            # Compute substitution cost
            cost = sm_normalized[state_i_idx, state_j_idx]
            
            if transindel in ["constant", "prob", "subcost"]:
                cost = otto * cost + (indels[i] + indels[j] - 
                                     stateindel[state_i_idx] - stateindel[state_j_idx])
            
            newsm[i, j] = cost
            newsm[j, i] = cost
    
    return newsm, indels, newalph
