"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqcompare.py
@Time    : 2026-02-10 21:11
@Desc    : Likelihood Ratio Test and Bayesian Information Criterion for comparing
           sets of sequences using dissimilarity measures.
           
           This module implements functions for comparing two groups of sequences
           by computing LRT (Likelihood Ratio Test) and BIC (Bayesian Information Criterion)
           statistics. These tests help determine if two groups of sequences are
           significantly different.
           
           Corresponds to TraMineRextras functions: seqCompare(), seqLRT(), seqBIC()
           
           Authors: Tim Liao (University of Illinois) and Anette Fasang (Humboldt University)
           Python implementation: Yuqi Liang
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import warnings
import time

from ..dissimilarity_measures.get_distance_matrix import get_distance_matrix
from ..tree_analysis.tree_utils import compute_pseudo_variance
from ..define_sequence_data import SequenceData


def compute_bayesian_information_criterion_test(
    seqdata: Union[pd.DataFrame, List[pd.DataFrame]],
    seqdata2: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    squared: str = "LRTonly",
    weighted: bool = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    """
    Compute Bayesian Information Criterion for comparing sequences.
    
    This is a convenience wrapper around compare_groups_overall() that returns only BIC statistics.
    
    **Corresponds to TraMineRextras function: `seqBIC()`**
    
    **TraMineRextras Equivalent:**
    ```r
    # In R (TraMineRextras package):
    seqBIC(seqdata, seqdata2 = NULL, group = NULL, set = NULL,
           s = 100, seed = 36963, squared = "LRTonly", 
           weighted = TRUE, method = "OM")
    ```
    
    Parameters
    ----------
    See seqcompare() for parameter descriptions.
    
    Returns
    -------
    np.ndarray
        Results matrix with BIC statistics and Bayes Factors
        
    See Also
    --------
    compare_groups_overall : Main function for complete comparison
    compute_likelihood_ratio_test : Compute LRT statistics only
    """
    return compare_groups_overall(
        seqdata=seqdata,
        seqdata2=seqdata2,
        group=group,
        set_var=set_var,
        s=s,
        seed=seed,
        stat="BIC",
        squared=squared,
        weighted=weighted,
        opt=opt,
        BFopt=BFopt,
        method=method,
        **kwargs
    )


def compute_likelihood_ratio_test(
    seqdata: Union[pd.DataFrame, List[pd.DataFrame]],
    seqdata2: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    squared: str = "LRTonly",
    weighted: bool = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    """
    Compute Likelihood Ratio Test for comparing sequences.
    
    This is a convenience wrapper around compare_groups_overall() that returns only LRT statistics.
    
    **Corresponds to TraMineRextras function: `seqLRT()`**
    
    **TraMineRextras Equivalent:**
    ```r
    # In R (TraMineRextras package):
    seqLRT(seqdata, seqdata2 = NULL, group = NULL, set = NULL,
           s = 100, seed = 36963, squared = "LRTonly", 
           weighted = TRUE, method = "OM")
    ```
    
    Parameters
    ----------
    See seqcompare() for parameter descriptions.
    
    Returns
    -------
    np.ndarray
        Results matrix with LRT statistics and p-values
        
    See Also
    --------
    compare_groups_overall : Main function for complete comparison
    compute_bayesian_information_criterion_test : Compute BIC statistics only
    """
    return compare_groups_overall(
        seqdata=seqdata,
        seqdata2=seqdata2,
        group=group,
        set_var=set_var,
        s=s,
        seed=seed,
        stat="LRT",
        squared=squared,
        weighted=weighted,
        opt=opt,
        BFopt=BFopt,
        method=method,
        **kwargs
    )


def compare_groups_overall(
    seqdata: Union[pd.DataFrame, SequenceData, List[pd.DataFrame], List[SequenceData]],
    seqdata2: Optional[Union[pd.DataFrame, SequenceData, List[pd.DataFrame], List[SequenceData]]] = None,
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    set_var: Optional[Union[np.ndarray, pd.Series]] = None,
    s: int = 100,
    seed: int = 36963,
    stat: str = "all",
    squared: Union[bool, str] = "LRTonly",
    weighted: Union[bool, str] = True,
    opt: Optional[int] = None,
    BFopt: Optional[int] = None,
    method: str = "OM",
    **kwargs
) -> np.ndarray:
    """
    Compare sets of sequences using LRT and BIC statistics (overall difference test).
    
    This function compares two groups of sequences by computing likelihood ratio test (LRT)
    statistics and Bayesian Information Criterion (BIC). The comparison can be done either
    between two separate sequence datasets (seqdata vs seqdata2) or between groups within
    a single dataset (using the group parameter).
    
    Unlike compare_groups_across_positions() which analyzes differences at each position,
    this function tests whether groups are significantly different overall across the
    entire sequence.
    
    **Corresponds to TraMineRextras function: `seqCompare()`**
    
    **TraMineRextras Equivalent:**
    ```r
    # In R (TraMineRextras package):
    seqCompare(seqdata, seqdata2 = NULL, group = NULL, set = NULL,
               s = 100, seed = 36963, stat = "all", 
               squared = "LRTonly", weighted = TRUE, method = "OM")
    ```
    
    Parameters
    ----------
    seqdata : pd.DataFrame or list of pd.DataFrame
        State sequence data. Can be:
        - A single DataFrame: sequences to compare (requires seqdata2 or group)
        - A list of DataFrames: multiple sequence sets to compare pairwise with seqdata2
        
    seqdata2 : pd.DataFrame or list of pd.DataFrame, optional
        Second set of sequences to compare against seqdata.
        If None, uses group parameter to split seqdata.
        Default: None
        
    group : np.ndarray or pd.Series, optional
        Grouping variable for splitting seqdata into two groups.
        Only used if seqdata2 is None.
        Currently supports only 2 groups.
        Default: None
        
    set_var : np.ndarray or pd.Series, optional
        Variable defining sets for stratified comparison.
        When provided, comparisons are done separately for each set level.
        Requires group to be specified.
        Default: None
        
    s : int, optional
        Sample size for bootstrap sampling.
        If s=0, no sampling is performed (uses all sequences).
        If s>0, performs bootstrap sampling with replacement.
        Default: 100
        
    seed : int, optional
        Random seed for reproducibility.
        Default: 36963
        
    stat : str, optional
        Which statistics to compute. Can be:
        - "LRT": Likelihood Ratio Test only
        - "BIC": Bayesian Information Criterion only
        - "all": Both LRT and BIC
        Default: "all"
        
    squared : bool or str, optional
        How to handle distance squaring:
        - True: Square distances before computing statistics
        - False: Use distances as-is
        - "LRTonly": Square for LRT computation only (power=2), use unsquared for BIC
        Default: "LRTonly"
        
    weighted : bool or str, optional
        Whether to use sequence weights:
        - True: Use weights with 'global' normalization
        - False: Ignore weights
        - 'by.group': Normalize weights within each group
        Default: True
        
    opt : int, optional
        Optimization strategy for distance computation:
        - 1: Compute distances for each bootstrap sample separately (slower but less memory)
        - 2: Compute full distance matrix once and subsample (faster but more memory)
        - None: Auto-select based on sample size (uses opt=1 if n1+n2 > 2*s, else opt=2)
        Default: None (auto)
        
    BFopt : int, optional
        Bayes Factor computation option when multiple samples:
        - 1: Average Bayes Factors across samples
        - 2: Compute Bayes Factor from averaged BIC
        - None: Return both versions
        Default: None (both)
        
    method : str, optional
        Distance computation method to use.
        Common methods: "OM", "LCS", "HAM", etc.
        See get_distance_matrix() for full list.
        Default: "OM"
        
    **kwargs
        Additional arguments passed to get_distance_matrix()
        (e.g., indel, sm, norm, etc.)
        
    Returns
    -------
    np.ndarray
        Results matrix with shape (G, nc) where:
        - G: Number of comparisons (1 if set_var is None, else number of set levels)
        - nc: Number of columns depending on stat parameter:
            - stat="LRT": 2 columns (LRT, p-value)
            - stat="BIC": 2-3 columns (Delta BIC, Bayes Factor, [Bayes Factor from Avg BIC])
            - stat="all": 4-5 columns (LRT, p-value, Delta BIC, Bayes Factor, [BF from Avg])
        
        Column names are added as array attributes.
        Row names correspond to set levels if set_var is provided.
        
    Notes
    -----
    - The LRT statistic tests the null hypothesis that two groups have the same
      sequence distribution against the alternative that they differ
    - Higher LRT values indicate greater differences between groups
    - The p-value is computed using chi-square distribution with 1 degree of freedom
    - BIC (Bayesian Information Criterion) = LRT - log(n)
    - Bayes Factor = exp(BIC/2), values > 1 indicate evidence for group differences
    - When s > 0, bootstrap sampling is used and statistics are averaged across samples
    - The 'LRTonly' squared option uses squared distances (power=2) for LRT computation
      but regular distances for BIC, following the original TraMineRextras implementation
      
    Computational Strategy:
    - opt=1: Good when sample size s is much smaller than total sequences (n1+n2)
             Computes distances separately for each bootstrap sample
    - opt=2: Good when s is close to total sequences or memory is not a constraint
             Computes full distance matrix once and subsamples from it
             
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sequenzo.compare_differences import seqcompare, seqLRT, seqBIC
    >>> 
    >>> # Create sample sequence data
    >>> seqdata = pd.DataFrame({
    ...     'pos1': ['A', 'A', 'B', 'B', 'A', 'A'],
    ...     'pos2': ['A', 'B', 'B', 'C', 'A', 'B'],
    ...     'pos3': ['B', 'B', 'C', 'C', 'B', 'B']
    ... })
    >>> 
    >>> # Define groups (2 groups required)
    >>> group = np.array([1, 1, 1, 2, 2, 2])
    >>> 
    >>> # Compare groups using LRT
    >>> lrt_result = seqLRT(seqdata, group=group, method="LCS", s=50)
    >>> print("LRT:", lrt_result[0, 0])
    >>> print("p-value:", lrt_result[0, 1])
    >>> 
    >>> # Compare using BIC
    >>> bic_result = seqBIC(seqdata, group=group, method="LCS", s=50)
    >>> print("Delta BIC:", bic_result[0, 0])
    >>> print("Bayes Factor:", bic_result[0, 1])
    >>> 
    >>> # Compare two separate sequence datasets
    >>> seq1 = seqdata.iloc[:3, :]
    >>> seq2 = seqdata.iloc[3:, :]
    >>> result = seqcompare(seq1, seq2, method="LCS", s=0)
    
    References
    ----------
    Liao, T. F., Bolano, D., Brzinsky-Fay, C., Cornwell, B., Fasang, A. E.,
    Helske, S., Piccarreta, R., Raab, M., Ritschard, G., Struffolino, E., and
    Studer, M. (2022). Sequence analysis: Its past, present, and future.
    Social Science Research, 107, 102772.
    
    Studer, M. and Ritschard, G. (2016). What matters in differences between
    life trajectories: A comparative review of sequence dissimilarity measures.
    Journal of the Royal Statistical Society: Series A, 179(2), 481-511.
    """
    # Start timing
    ptime_begin = time.time()
    
    # Validate inputs
    if seqdata2 is None and group is None:
        raise ValueError("[!] 'seqdata2' and 'group' cannot both be None!")
    
    if set_var is not None and group is None:
        raise ValueError("[!] 'set_var' not None while 'group' is None!")
    
    # Handle weighted parameter
    if isinstance(weighted, str):
        if weighted != 'by.group':
            raise ValueError("[!] weighted must be logical or 'by.group'")
        weight_by = weighted
        weighted = True
    else:
        weight_by = 'global'
    
    # Handle squared parameter for LRT power
    if isinstance(squared, bool):
        LRTpow = 1
    else:
        if squared != "LRTonly":
            raise ValueError("[!] squared must be logical or 'LRTonly'")
        LRTpow = 2
        squared = False
    
    # Check if inputs are sequence objects
    is1_seqdata = isinstance(seqdata, (pd.DataFrame, SequenceData))
    is2_seqdata = isinstance(seqdata2, (pd.DataFrame, SequenceData)) if seqdata2 is not None else False
    
    # Handle list inputs
    if isinstance(seqdata, list):
        if is2_seqdata or (seqdata2 is not None and not isinstance(seqdata2, list)) or len(seqdata) != len(seqdata2):
            raise ValueError(
                "[!] When 'seqdata' is a list, seqdata2 must be a list of same length"
            )
        else:
            # Verify all elements are DataFrames or SequenceData
            for i, (sd1, sd2) in enumerate(zip(seqdata, seqdata2)):
                if not isinstance(sd1, (pd.DataFrame, SequenceData)) or not isinstance(sd2, (pd.DataFrame, SequenceData)):
                    raise TypeError(
                        f"[!] At least one element of the seqdata lists at index {i} "
                        "is not a DataFrame or SequenceData!"
                    )
    elif not is1_seqdata:
        raise TypeError(
            "[!] If not a list, 'seqdata' must be a DataFrame or SequenceData object"
        )
    elif seqdata2 is not None and not is2_seqdata:
        raise TypeError(
            "[!] If not a list, 'seqdata2' must be a DataFrame or SequenceData object"
        )
    
    # Convert SequenceData to DataFrame for processing (only if not a list)
    if isinstance(seqdata, list):
        seqdata_df = None  # Will be handled in list processing
        seqdata_original = seqdata
    elif isinstance(seqdata, SequenceData):
        seqdata_df = seqdata.data[seqdata.time].copy()
        seqdata_original = seqdata  # Keep reference for weights, etc.
    elif isinstance(seqdata, pd.DataFrame):
        seqdata_df = seqdata.copy()
        seqdata_original = seqdata
    else:
        seqdata_df = seqdata
        seqdata_original = seqdata
    
    if seqdata2 is not None:
        if isinstance(seqdata2, SequenceData):
            seqdata2_df = seqdata2.data[seqdata2.time].copy()
        elif isinstance(seqdata2, pd.DataFrame):
            seqdata2_df = seqdata2.copy()
        else:
            seqdata2_df = seqdata2
    else:
        seqdata2_df = None
    
    # Validate stat parameter
    valid_stats = ["LRT", "BIC", "all"]
    if not all(s in valid_stats for s in ([stat] if isinstance(stat, str) else stat)):
        raise ValueError(
            f"[!] Bad stat value, must be one of {', '.join(valid_stats)}"
        )
    
    # Determine which statistics to compute
    if stat == "all":
        is_LRT = is_BIC = True
    else:
        is_LRT = "LRT" in ([stat] if isinstance(stat, str) else stat)
        is_BIC = "BIC" in ([stat] if isinstance(stat, str) else stat)
    
    # Prepare sequence lists
    if not is1_seqdata:
        # Already lists
        seq1 = seqdata
        seq2 = seqdata2
    elif is1_seqdata and seqdata2 is not None:
        # Single DataFrames/SequenceData, wrap in lists
        seq1 = [seqdata_df]
        seq2 = [seqdata2_df]
    elif is1_seqdata and group is not None:
        # Split by group variable
        gvar = np.asarray(group)
        
        # Handle set variable
        if set_var is not None:
            setvar = np.asarray(set_var)
            inotna = np.where(~(pd.isna(gvar) | pd.isna(setvar)))[0]
            setvar = setvar[inotna]
            setvar = pd.Categorical(setvar)
            lev_set = setvar.categories.tolist()
        else:
            inotna = np.where(~pd.isna(gvar))[0]
        
        # Report removed sequences
        n_removed = len(gvar) - len(inotna)
        if n_removed > 0:
            print(f"[!!] {n_removed} sequences removed because of NA values "
                  "of the grouping variable(s)")
        
        # Filter data
        gvar = gvar[inotna]
        gvar = pd.Categorical(gvar)
        lev_g = gvar.categories.tolist()
        
        # Validate number of groups
        if len(lev_g) == 1:
            raise ValueError("[!] There is only one group among valid cases!")
        if len(lev_g) > 2:
            raise ValueError("[!] Currently seqcompare supports only 2 groups!")
        
        seqdata_filtered = seqdata_df.iloc[inotna, :]
        
        # Split into two sequence lists
        seq1 = []
        seq2 = []
        
        if set_var is None:
            seq1_df = seqdata_filtered[gvar == lev_g[0]]
            seq2_df = seqdata_filtered[gvar == lev_g[1]]
            seq1.append(seq1_df)
            seq2.append(seq2_df)
        else:
            setvar_filtered = setvar
            for i, lev in enumerate(lev_set):
                mask1 = (gvar == lev_g[0]) & (setvar_filtered == lev)
                mask2 = (gvar == lev_g[1]) & (setvar_filtered == lev)
                seq1.append(seqdata_filtered[mask1])
                seq2.append(seqdata_filtered[mask2])
    
    # Prepare samples
    G = len(seq1)
    n = np.zeros((G, 2), dtype=int)
    seq_a = []
    seq_b = []
    
    for i in range(G):
        n1 = len(seq1[i])
        n2 = len(seq2[i])
        
        if n1 >= n2:
            n[i, 0] = n1
            n[i, 1] = n2
            seq_a.append(seq1[i])
            seq_b.append(seq2[i])
        else:
            n[i, 0] = n2
            n[i, 1] = n1
            seq_a.append(seq2[i])
            seq_b.append(seq1[i])
    
    # Compute sampling parameters
    n_n = n.min(axis=1)  # minimum group size for each comparison
    
    if s > 0:
        m_n = n.max(axis=1)  # maximum group size
        f_n1 = np.floor(s / m_n).astype(int)
        ff_n1 = np.maximum(1, f_n1)
        r_n1 = np.where(s < m_n, s - (m_n % s), s - f_n1 * m_n)
        k_n = np.floor((ff_n1 * m_n + r_n1) / n_n).astype(int)
        k_n[pd.isna(k_n)] = 0
        r_n2 = (ff_n1 * m_n + r_n1) - k_n * n_n
        r_n2[pd.isna(r_n2)] = 0
        
        # Error checking
        if np.any(m_n < r_n1):
            ii = np.where(m_n < r_n1)[0]
            raise ValueError(
                f"[!] rest r_n1 values greater than max m_n for i={ii}, s={s}"
            )
        if np.any(n_n < r_n2):
            ii = np.where(n_n < r_n2)[0]
            raise ValueError(
                f"[!] rest r_n2 values greater than min n_n for i={ii}, s={s}"
            )
    
    # Initialize results
    nc = 4 if (is_LRT and is_BIC) else 2
    Results = np.full((G, nc), np.nan)
    multsple = False
    
    # Process each comparison
    for i in range(G):
        if n_n[i] > 0:
            if s == 0:
                # No sampling - use all sequences
                r1 = np.arange(len(seq_a[i]))
                r2 = np.arange(len(seq_b[i])) + len(seq_a[i])
                
                # Combine sequences
                combined_seqs_df = pd.concat([seq_a[i], seq_b[i]], ignore_index=True)
                
                # Get weights
                weights_a = np.ones(len(seq_a[i]))
                weights_b = np.ones(len(seq_b[i]))
                if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, 'weights') and weighted:
                    # Extract weights for filtered sequences
                    if group is not None:
                        mask_a = gvar == lev_g[0]
                        mask_b = gvar == lev_g[1]
                        weights_a = seqdata_original.weights[inotna][mask_a]
                        weights_b = seqdata_original.weights[inotna][mask_b]
                weights = np.concatenate([weights_a, weights_b])
                
                # Create temporary SequenceData
                temp_states = None
                if isinstance(seqdata_original, SequenceData):
                    temp_states = seqdata_original.states
                else:
                    temp_states = sorted(combined_seqs_df.stack().dropna().unique().tolist())
                
                combined_seqs = SequenceData(
                    combined_seqs_df,
                    time=list(combined_seqs_df.columns),
                    states=temp_states,
                    weights=weights
                )
                
                # Compute distances
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    diss = get_distance_matrix(
                        seqdata=combined_seqs,
                        method=method,
                        weighted=weighted,
                        **kwargs
                    )
                
                # Convert diss to numpy array if it's a DataFrame
                if isinstance(diss, pd.DataFrame):
                    diss = diss.values
                else:
                    diss = np.asarray(diss)
                
                # Compute comparison statistics
                Results[i, :] = _seqxcomp(
                    r1, r2, diss, weights,
                    is_LRT=is_LRT, is_BIC=is_BIC,
                    squared=squared, weighted=weighted,
                    weight_by=weight_by, LRTpow=LRTpow
                )
            else:
                # Sampling
                np.random.seed(seed)
                
                # Generate sample indices
                r_s1_flat = np.concatenate([
                    np.random.permutation(np.repeat(np.arange(m_n[i]), ff_n1[i])),
                    np.random.choice(m_n[i], r_n1[i], replace=False)
                ])
                r_s2_flat = np.concatenate([
                    np.random.permutation(np.repeat(np.arange(n_n[i]), k_n[i])),
                    np.random.choice(n_n[i], r_n2[i], replace=False)
                ])
                
                # Reshape into samples
                # Ensure we have enough elements to reshape
                num_samples_1 = len(r_s1_flat) // s
                num_samples_2 = len(r_s2_flat) // s
                r_s1 = r_s1_flat[:num_samples_1 * s].reshape(num_samples_1, s)
                r_s2 = r_s2_flat[:num_samples_2 * s].reshape(num_samples_2, s)
                
                # Determine optimization strategy
                if opt is None:
                    opt_i = 1 if (len(seq_a[i]) + len(seq_b[i])) > 2 * s else 2
                else:
                    opt_i = opt
                
                # Precompute full distance matrix if opt=2
                if opt_i == 2:
                    combined_seqs_df = pd.concat([seq_a[i], seq_b[i]], ignore_index=True)
                    
                    # Get weights (similar to s=0 case)
                    weights_a = np.ones(len(seq_a[i]))
                    weights_b = np.ones(len(seq_b[i]))
                    if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, 'weights') and weighted:
                        if group is not None:
                            mask_a = gvar == lev_g[0]
                            mask_b = gvar == lev_g[1]
                            weights_a = seqdata_original.weights[inotna][mask_a]
                            weights_b = seqdata_original.weights[inotna][mask_b]
                    weights = np.concatenate([weights_a, weights_b])
                    
                    # Create temporary SequenceData
                    temp_states = None
                    if isinstance(seqdata_original, SequenceData):
                        temp_states = seqdata_original.states
                    else:
                        temp_states = sorted(combined_seqs_df.stack().dropna().unique().tolist())
                    
                    combined_seqs = SequenceData(
                        combined_seqs_df,
                        time=list(combined_seqs_df.columns),
                        states=temp_states,
                        weights=weights
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        diss_full = get_distance_matrix(
                            seqdata=combined_seqs,
                            method=method,
                            weighted=weighted,
                            **kwargs
                        )
                    # Convert to numpy array
                    if isinstance(diss_full, pd.DataFrame):
                        diss_full = diss_full.values
                    else:
                        diss_full = np.asarray(diss_full)
                
                multsple = r_s1.shape[0] > 1 or multsple
                
                # Process each sample
                t = np.zeros((r_s1.shape[0], nc))
                
                for j in range(r_s1.shape[0]):
                    if opt_i == 2:
                        # Use precomputed distance matrix
                        # r1 and r2 are indices into the full distance matrix
                        r1 = r_s1[j, :]
                        r2 = r_s2[j, :] + len(seq_a[i])
                        diss = diss_full
                        # Get weights for sampled sequences
                        # Extract weights for the sampled indices
                        if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, 'weights') and weighted:
                            if group is not None:
                                mask_a = gvar == lev_g[0]
                                mask_b = gvar == lev_g[1]
                                # Get weights for all sequences in groups
                                weights_all_a = seqdata_original.weights[inotna][mask_a]
                                weights_all_b = seqdata_original.weights[inotna][mask_b]
                                # Verify that weights array sizes match sequence array sizes
                                if len(weights_all_a) != len(seq_a[i]):
                                    raise ValueError(
                                        f"Weights array size mismatch: weights_all_a has {len(weights_all_a)} elements, "
                                        f"but seq_a[i] has {len(seq_a[i])} rows. "
                                        f"m_n[i]={m_n[i] if i < len(m_n) else 'N/A'}, "
                                        f"mask_a sum={np.sum(mask_a)}"
                                    )
                                if len(weights_all_b) != len(seq_b[i]):
                                    raise ValueError(
                                        f"Weights array size mismatch: weights_all_b has {len(weights_all_b)} elements, "
                                        f"but seq_b[i] has {len(seq_b[i])} rows. "
                                        f"n_n[i]={n_n[i] if i < len(n_n) else 'N/A'}, "
                                        f"mask_b sum={np.sum(mask_b)}"
                                    )
                                # Select weights for sampled sequences
                                # r_s1[j, :] are indices into seq_a[i], which should correspond to positions in weights_all_a
                                r_s1_flat = np.asarray(r_s1[j, :]).flatten()
                                r_s2_flat = np.asarray(r_s2[j, :]).flatten()
                                # Ensure indices are within bounds
                                if len(weights_all_a) > 0 and np.any(r_s1_flat >= len(weights_all_a)):
                                    max_idx = np.max(r_s1_flat)
                                    raise ValueError(
                                        f"Sampled index {max_idx} out of bounds for weights_all_a of size {len(weights_all_a)}. "
                                        f"seq_a[i] size: {len(seq_a[i])}, m_n[i]={m_n[i] if i < len(m_n) else 'N/A'}"
                                    )
                                if len(weights_all_b) > 0 and np.any(r_s2_flat >= len(weights_all_b)):
                                    max_idx = np.max(r_s2_flat)
                                    raise ValueError(
                                        f"Sampled index {max_idx} out of bounds for weights_all_b of size {len(weights_all_b)}. "
                                        f"seq_b[i] size: {len(seq_b[i])}, n_n[i]={n_n[i] if i < len(n_n) else 'N/A'}"
                                    )
                                weights_a = weights_all_a[r_s1_flat]
                                weights_b = weights_all_b[r_s2_flat]
                            else:
                                weights_a = np.ones(len(r_s1[j, :]))
                                weights_b = np.ones(len(r_s2[j, :]))
                        else:
                            weights_a = np.ones(len(r_s1[j, :]))
                            weights_b = np.ones(len(r_s2[j, :]))
                        weights = np.concatenate([weights_a, weights_b])
                    else:
                        # Compute distance matrix for this sample
                        # Convert array indices to list for iloc
                        # r_s1[j, :] returns a 1D array, convert to list
                        indices_a = np.asarray(r_s1[j, :]).flatten().tolist()
                        indices_b = np.asarray(r_s2[j, :]).flatten().tolist()
                        seqA_df = seq_a[i].iloc[indices_a, :]
                        seqB_df = seq_b[i].iloc[indices_b, :]
                        seqAB_df = pd.concat([seqA_df, seqB_df], ignore_index=True)
                        
                        # Get weights
                        wA = np.ones(len(seqA_df))
                        wB = np.ones(len(seqB_df))
                        if isinstance(seqdata_original, SequenceData) and hasattr(seqdata_original, 'weights') and weighted:
                            # Extract weights for sampled sequences
                            if group is not None:
                                mask_a = gvar == lev_g[0]
                                mask_b = gvar == lev_g[1]
                                # Get indices where mask is True, then select sampled ones
                                mask_a_indices = np.where(mask_a)[0]
                                mask_b_indices = np.where(mask_b)[0]
                                # Use the sampled indices to select from mask indices
                                # Ensure r_s1[j, :] is 1D array
                                r_s1_flat = np.asarray(r_s1[j, :]).flatten()
                                r_s2_flat = np.asarray(r_s2[j, :]).flatten()
                                sampled_indices_a = mask_a_indices[r_s1_flat]
                                sampled_indices_b = mask_b_indices[r_s2_flat]
                                wA = seqdata_original.weights[inotna][sampled_indices_a]
                                wB = seqdata_original.weights[inotna][sampled_indices_b]
                        weights = np.concatenate([wA, wB])
                        
                        # Create temporary SequenceData
                        temp_states = None
                        if isinstance(seqdata_original, SequenceData):
                            temp_states = seqdata_original.states
                        else:
                            temp_states = sorted(seqAB_df.stack().dropna().unique().tolist())
                        
                        seqAB = SequenceData(
                            seqAB_df,
                            time=list(seqAB_df.columns),
                            states=temp_states,
                            weights=weights
                        )
                        
                        r1 = np.arange(len(r_s1[j, :]))
                        r2 = np.arange(len(r_s2[j, :])) + len(r_s1[j, :])
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            diss = get_distance_matrix(
                                seqdata=seqAB,
                                method=method,
                                weighted=weighted,
                                **kwargs
                            )
                        # Convert to numpy array
                        if isinstance(diss, pd.DataFrame):
                            diss = diss.values
                        else:
                            diss = np.asarray(diss)
                    
                    # Compute statistics for this sample
                    t[j, :] = _seqxcomp(
                        r1, r2, diss, weights,
                        is_LRT=is_LRT, is_BIC=is_BIC,
                        squared=squared, weighted=weighted,
                        weight_by=weight_by, LRTpow=LRTpow
                    )
                
                # Average across samples
                Results[i, :] = t.mean(axis=0)
    
    # Build column names
    colnames = []
    if is_LRT:
        colnames.extend(["LRT", "p-value"])
    if is_BIC:
        if BFopt is None and multsple:
            # Add Bayes Factor from averaged BIC
            BF2 = np.exp(Results[:, nc - 1] / 2)
            Results = np.column_stack([Results, BF2])
            colnames.extend(["Delta BIC", "Bayes Factor (Avg)", "Bayes Factor (From Avg BIC)"])
        elif BFopt == 1 and multsple:
            colnames.extend(["Delta BIC", "Bayes Factor (Avg)"])
        elif BFopt == 2 and multsple:
            BF2 = np.exp(Results[:, nc - 1] / 2)
            Results[:, nc] = BF2
            colnames.extend(["Delta BIC", "Bayes Factor (From Avg BIC)"])
        else:
            colnames.extend(["Delta BIC", "Bayes Factor"])
    
    # Add metadata to results
    Results = pd.DataFrame(Results, columns=colnames)
    
    # Add row names if set_var was used
    if set_var is not None:
        Results.index = lev_set
    
    # Print elapsed time
    ptime_end = time.time()
    elapsed = ptime_end - ptime_begin
    print(f"elapsed time: {elapsed:.3f} seconds")
    
    return Results.values


def _seqxcomp(
    r1: np.ndarray,
    r2: np.ndarray,
    diss: np.ndarray,
    weights: np.ndarray,
    is_LRT: bool,
    is_BIC: bool,
    squared: bool,
    weighted: bool,
    weight_by: str,
    LRTpow: int
) -> np.ndarray:
    """
    Compute comparison statistics for a single sample.
    
    This is an internal helper function that computes LRT and BIC statistics
    for comparing two groups based on their distance matrix.
    
    Parameters
    ----------
    r1 : np.ndarray
        Indices of group 1 sequences
    r2 : np.ndarray
        Indices of group 2 sequences
    diss : np.ndarray
        Distance matrix
    weights : np.ndarray
        Sequence weights
    is_LRT : bool
        Whether to compute LRT statistics
    is_BIC : bool
        Whether to compute BIC statistics
    squared : bool
        Whether to square distances
    weighted : bool
        Whether to use weights
    weight_by : str
        Weight normalization method ('global' or 'by.group')
    LRTpow : int
        Power to use for LRT computation (1 or 2)
        
    Returns
    -------
    np.ndarray
        Array of statistics (LRT, p-value, BIC, BF) depending on is_LRT and is_BIC
    """
    # Basic counts
    n1 = len(r1)
    n2 = len(r2)
    n0 = n1 + n2
    
    # Handle weights
    weighted = weighted and weights is not None
    
    if weighted:
        w1 = weights[r1]
        w2 = weights[r2]
        w = np.concatenate([w1, w2])
        
        # Normalize weights if needed
        if weight_by == 'by.group':
            w1 = n1 / w1.sum() * w1
            w2 = n2 / w2.sum() * w2
            w = np.concatenate([w1, w2])
        
        nw = w.sum()
        nw1 = w1.sum()
        nw2 = w2.sum()
    else:
        nw = n0
        nw1 = n1
        nw2 = n2
        w = np.ones(n0)
        w1 = np.ones(n1)
        w2 = np.ones(n2)
    
    # Ensure diss is numpy array (not DataFrame)
    if isinstance(diss, pd.DataFrame):
        diss_array = diss.values
    else:
        diss_array = np.asarray(diss)
    
    # Compute distances to center
    r_combined = np.concatenate([r1, r2])
    dist_S = _disscenter(diss_array[np.ix_(r_combined, r_combined)], weights=w, squared=squared)
    dist_S1 = _disscenter(diss_array[np.ix_(r1, r1)], weights=w1, squared=squared)
    dist_S2 = _disscenter(diss_array[np.ix_(r2, r2)], weights=w2, squared=squared)
    
    # Compute sum of squares with appropriate power
    SS = (w * (dist_S ** LRTpow)).sum()
    SS1 = (w1 * (dist_S1 ** LRTpow)).sum()
    SS2 = (w2 * (dist_S2 ** LRTpow)).sum()
    
    # Compute LRT
    LRT = n0 * (np.log(SS / n0) - np.log((SS1 + SS2) / n0))
    
    res = []
    
    if is_LRT:
        # Compute p-value using chi-square distribution
        from scipy.stats import chi2
        p_LRT = chi2.sf(LRT, df=1)
        res.extend([LRT, p_LRT])
    
    if is_BIC:
        # Compute BIC and Bayes Factor
        BIC = LRT - 1 * np.log(n0)
        BF = np.exp(BIC / 2)
        res.extend([BIC, BF])
    
    return np.array(res)


def _disscenter(
    diss: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> np.ndarray:
    """
    Compute distance to center for each element.
    
    This is a simplified version of disscenter from TraMineR that computes
    the distance from each sequence to the center (weighted centroid) of the group.
    
    Parameters
    ----------
    diss : np.ndarray
        Distance matrix (square symmetric)
    weights : np.ndarray, optional
        Weights for each element
    squared : bool, optional
        Whether to square distances
        
    Returns
    -------
    np.ndarray
        Distance of each element to the center
    """
    if squared:
        diss = diss ** 2
    
    n = diss.shape[0]
    
    if weights is None:
        weights = np.ones(n)
    
    weights = np.asarray(weights, dtype=np.float64)
    
    # Compute weighted distance to center
    # For each element i, compute weighted sum of distances to all elements
    # Then subtract the weighted mean to center it
    dist_center = np.zeros(n)
    
    for i in range(n):
        # Weighted sum of distances from i to all other elements
        weighted_sum = (weights * diss[i, :]).sum()
        dist_center[i] = weighted_sum
    
    # Center by subtracting weighted mean
    weighted_mean = (weights * dist_center).sum() / weights.sum()
    dist_center = dist_center - weighted_mean / 2
    
    return dist_center
