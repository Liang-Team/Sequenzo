"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqtree.py
@Time    : 2026-02-09 17:01
@Desc    : Sequence regression tree - wrapper for distance tree analysis on sequences.

           Corresponds to TraMineR function: seqtree()
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix

from .disstree import build_distance_tree


def build_sequence_tree(
    seqdata: SequenceData,
    predictors: pd.DataFrame,
    distance_matrix: Optional[np.ndarray] = None,
    distance_method: str = "LCS",
    distance_params: Optional[Dict[str, Any]] = None,
    weighted: bool = True,
    min_size: Union[float, int] = 0.05,
    max_depth: int = 5,
    R: int = 1000,
    pval: float = 0.01,
    weight_permutation: str = "replicate",
    squared: bool = False,
    first_split: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a regression tree for sequence data.
    
    This is a simplified interface for applying distance-based tree analysis
    to state sequence objects. It automatically computes the distance matrix
    if not provided, and extracts weights from the sequence data.
    
    **Corresponds to TraMineR function: `seqtree()`**
    
    Parameters
    ----------
    seqdata : SequenceData
        State sequence object created with SequenceData()
        
    predictors : pd.DataFrame
        DataFrame with covariates (predictor variables) for each sequence.
        Must have same number of rows as sequences in seqdata.
        
    distance_matrix : np.ndarray, optional
        Pre-computed distance matrix. If None, will be computed using
        distance_method and distance_params.
        Default: None
        
    distance_method : str, optional
        Distance measure to use if distance_matrix is None.
        Options: "OM", "LCS", "HAM", "DHD", etc.
        Default: "LCS"
        
    distance_params : dict, optional
        Additional parameters for distance computation.
        Will be passed to get_distance_matrix().
        Default: None
        
    weighted : bool, optional
        Whether to use weights from seqdata.
        Default: True
        
    min_size : float or int, optional
        Minimum number of cases in a node. If float < 1, treated as proportion.
        Default: 0.05 (5% of total)
        
    max_depth : int, optional
        Maximum depth of the tree.
        Default: 5
        
    R : int, optional
        Number of permutations for significance testing.
        Default: 1000
        
    pval : float, optional
        Maximum allowed p-value for a split to be retained.
        Default: 0.01
        
    weight_permutation : str, optional
        Method for handling weights in permutation tests.
        Default: "replicate"
        
    squared : bool, optional
        If True, square the distance matrix before analysis.
        Default: False
        
    first_split : str, optional
        Name of variable to force as the first split.
        Default: None
        
    Returns
    -------
    dict
        A dictionary containing tree structure and metadata (same format as disstree).
        The result is a 'seqtree' object which extends 'disstree'.
        
    Examples
    --------
    >>> from sequenzo import SequenceData, load_dataset
    >>> from sequenzo.tree_analysis import build_sequence_tree
    >>> 
    >>> # Load data
    >>> df = load_dataset('dyadic_children')
    >>> time_list = sorted([c for c in df.columns if str(c).isdigit()], key=int)
    >>> seqdata = SequenceData(df.head(20), time=time_list, states=[1,2,3,4,5,6])
    >>> 
    >>> # Create predictors
    >>> predictors = pd.DataFrame({'group': ['A']*10 + ['B']*10})
    >>> 
    >>> # Build tree
    >>> tree = build_sequence_tree(seqdata, predictors, R=10, pval=0.1)
    >>> print(f"Number of leaves: {tree['fitted']['(fitted)'].nunique()}")
    
    References
    ----------
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011).
    Discrepancy analysis of state sequences.
    Sociological Methods and Research, Vol. 40(3), 471-510.
    """
    # Validate seqdata
    if not isinstance(seqdata, SequenceData):
        raise ValueError(
            "[!] 'seqdata' must be a SequenceData object. "
            "Use SequenceData() to create one."
        )
    
    n_seq = seqdata.seqdata.shape[0]
    
    # Validate predictors
    if len(predictors) != n_seq:
        raise ValueError(
            f"[!] Number of rows in 'predictors' ({len(predictors)}) "
            f"must match number of sequences ({n_seq})"
        )
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        if distance_params is None:
            distance_params = {}
        
        distance_params['method'] = distance_method
        distance_params['norm'] = distance_params.get('norm', 'auto')
        
        print(f"[>] Computing distance matrix using method '{distance_method}'...")
        distance_matrix = get_distance_matrix(seqdata=seqdata, **distance_params)
        
        if isinstance(distance_matrix, pd.DataFrame):
            distance_matrix = distance_matrix.values
    
    # Extract weights
    weights = None
    if weighted:
        if hasattr(seqdata, 'weights') and seqdata.weights is not None:
            weights = seqdata.weights
        else:
            weights = np.ones(n_seq, dtype=np.float64)
    
    # Build tree using disstree
    tree_result = build_distance_tree(
        distance_matrix=distance_matrix,
        predictors=predictors,
        weights=weights,
        min_size=min_size,
        max_depth=max_depth,
        R=R,
        pval=pval,
        weight_permutation=weight_permutation,
        squared=squared,
        first_split=first_split
    )
    
    # Mark as seqtree (extends disstree)
    tree_result['info']['method'] = 'seqtree'
    tree_result['seqdata'] = seqdata  # Store reference to original sequence data
    
    return tree_result
