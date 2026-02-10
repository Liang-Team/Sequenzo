"""
@Author  : Yuqi Liang 梁彧祺
@File    : tree_helpers.py
@Time    : 2026-02-10 08:31
@Desc    : Helper functions for tree analysis (leaf membership, rules, assignment).

           Corresponds to TraMineR functions: disstreeleaf(), disstree.get.rules(), disstree.assign()
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any
import re

from .tree_node import DissTreeNode


def get_leaf_membership(
    tree: Dict[str, Any],
    label: bool = False,
    collapse: str = ", "
) -> Union[np.ndarray, pd.Series]:
    """
    Get terminal node (leaf) membership for each sequence.
    
    Returns the ID of the terminal node that each sequence belongs to.
    Optionally returns human-readable labels describing the path to each leaf.
    
    **Corresponds to TraMineR function: `disstreeleaf()`**
    
    Parameters
    ----------
    tree : dict
        Tree object returned by build_distance_tree() or build_sequence_tree()
        
    label : bool, optional
        If True, return human-readable labels instead of node IDs.
        Labels describe the path from root to leaf (e.g., "gender=M & age<=25").
        Default: False
        
    collapse : str, optional
        String used to collapse multiple conditions in labels.
        Default: ", "
        
    Returns
    -------
    np.ndarray or pd.Series
        If label=False: Array of leaf node IDs (integers)
        If label=True: Series of leaf labels (strings)
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import build_distance_tree, get_leaf_membership
    >>> 
    >>> # Build tree (example)
    >>> tree = build_distance_tree(...)
    >>> 
    >>> # Get leaf IDs
    >>> leaf_ids = get_leaf_membership(tree)
    >>> print(f"Number of leaves: {len(np.unique(leaf_ids))}")
    >>> 
    >>> # Get labeled leaves
    >>> leaf_labels = get_leaf_membership(tree, label=True)
    >>> print(leaf_labels.head())
    """
    root = tree['root']
    n_total = len(tree['data'])
    
    if label:
        return _get_leaf_labels(root, tree, collapse)
    else:
        return _get_leaf_ids(root, n_total)


def _get_leaf_ids(root: DissTreeNode, n_total: int) -> np.ndarray:
    """Get leaf node IDs for each sequence."""
    memberships = np.full(n_total, -1, dtype=np.int32)
    _assign_leaf_ids_recursive(root, memberships)
    return memberships


def _assign_leaf_ids_recursive(node: DissTreeNode, memberships: np.ndarray):
    """Recursively assign leaf node IDs."""
    if node.is_terminal:
        memberships[node.info['ind']] = node.id
    else:
        _assign_leaf_ids_recursive(node.kids[0], memberships)
        _assign_leaf_ids_recursive(node.kids[1], memberships)


def _get_leaf_labels(
    root: DissTreeNode,
    tree: Dict[str, Any],
    collapse: str
) -> pd.Series:
    """Get human-readable labels for leaf nodes."""
    # Build label dictionary
    label_dict = {}
    _build_label_dict(root, tree, label_dict, [], collapse)
    
    # Get leaf IDs
    leaf_ids = _get_leaf_ids(root, len(tree['data']))
    
    # Map IDs to labels
    labels = [label_dict.get(int(node_id), f"Node_{node_id}") for node_id in leaf_ids]
    
    return pd.Series(labels, index=tree['data'].index)


def _build_label_dict(
    node: DissTreeNode,
    tree: Dict[str, Any],
    label_dict: Dict[int, str],
    current_path: List[str],
    collapse: str
):
    """Recursively build label dictionary."""
    if node.is_terminal:
        label_dict[node.id] = " & ".join(current_path) if current_path else "Root"
    else:
        split = node.split
        var_name = tree['data'].columns[split.varindex]
        
        # Format split condition
        if split.breaks is not None:
            # Continuous variable
            left_label = f"{var_name} <= {split.breaks:.2f}"
            right_label = f"{var_name} > {split.breaks:.2f}"
        else:
            # Categorical variable
            left_groups = []
            right_groups = []
            for i, idx in enumerate(split.index):
                if idx == 1:
                    left_groups.append(split.labels[i])
                elif idx == 2:
                    right_groups.append(split.labels[i])
            
            left_label = f"{var_name} in [{collapse.join(left_groups)}]"
            right_label = f"{var_name} in [{collapse.join(right_groups)}]"
        
        # Handle missing values
        if split.naGroup is not None:
            if split.naGroup == 1:
                left_label += " (with NA)"
            else:
                right_label += " (with NA)"
        
        # Recursively process children
        _build_label_dict(
            node.kids[0], tree, label_dict,
            current_path + [left_label], collapse
        )
        _build_label_dict(
            node.kids[1], tree, label_dict,
            current_path + [right_label], collapse
        )


def get_classification_rules(
    tree: Dict[str, Any],
    collapse: str = "; "
) -> List[str]:
    """
    Get classification rules as Python-compatible condition strings.
    
    Returns a list of rules, where each rule is a string that can be evaluated
    to determine if a sequence belongs to a particular leaf node.
    
    **Corresponds to TraMineR function: `disstree.get.rules()`**
    
    Parameters
    ----------
    tree : dict
        Tree object returned by build_distance_tree() or build_sequence_tree()
        
    collapse : str, optional
        String used to separate values in categorical conditions.
        Default: "; "
        
    Returns
    -------
    List[str]
        List of rule strings. Each rule is a Python expression that evaluates
        to True if a sequence matches that leaf node.
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import build_sequence_tree, get_classification_rules
    >>> 
    >>> # Build tree
    >>> tree = build_sequence_tree(...)
    >>> 
    >>> # Get rules
    >>> rules = get_classification_rules(tree)
    >>> print(f"Number of rules: {len(rules)}")
    >>> print(rules[0])  # Print first rule
    """
    leaf_labels = get_leaf_membership(tree, label=True, collapse=collapse)
    unique_labels = leaf_labels.unique()
    
    # Convert labels to Python conditions
    rules = []
    for label in unique_labels:
        # Convert label format to Python condition
        # Example: "gender in [M] & age <= 25.00" -> "(gender in ['M']) & (age <= 25.0)"
        rule = _label_to_python_condition(label, tree['data'].columns, collapse)
        rules.append(rule)
    
    return rules


def _label_to_python_condition(
    label: str,
    column_names: pd.Index,
    collapse: str
) -> str:
    """
    Convert human-readable label to Python-evaluable condition.
    
    This function converts labels like "gender in [M] & age <= 25.00"
    to Python expressions like "(gender in ['M']) & (age <= 25.0)".
    
    **Corresponds to TraMineR function: `disstree.get.rules()` conversion logic**
    
    Parameters
    ----------
    label : str
        Human-readable label from tree path (e.g., "gender in [M] & age <= 25.00")
    column_names : pd.Index
        Column names from predictors DataFrame
    collapse : str
        Separator used in categorical conditions (e.g., "; ")
        
    Returns
    -------
    str
        Python-evaluable condition string
    """
    # Split by " & " to get individual conditions
    conditions = label.split(" & ")
    python_conditions = []
    
    for cond in conditions:
        cond = cond.strip()
        
        # Handle categorical: "var in [val1; val2]" -> "var in ['val1', 'val2']"
        # Match pattern: variable_name in [values]
        cat_match = re.match(r'^(\w+)\s+in\s+\[(.+)\]', cond)
        if cat_match:
            var_name = cat_match.group(1)
            values_str = cat_match.group(2)
            
            # Split values by collapse separator
            values = [v.strip() for v in values_str.split(collapse)]
            
            # Convert to Python list of strings
            values_python = "', '".join(values)
            python_cond = f"({var_name} in ['{values_python}'])"
            python_conditions.append(python_cond)
            continue
        
        # Handle continuous: "var <= value" or "var > value"
        # Match patterns: variable_name <= value or variable_name > value
        cont_match = re.match(r'^(\w+)\s+(<=|>)\s+([0-9.]+)', cond)
        if cont_match:
            var_name = cont_match.group(1)
            operator = cont_match.group(2)
            value = cont_match.group(3)
            
            # Convert to Python comparison
            python_cond = f"({var_name} {operator} {value})"
            python_conditions.append(python_cond)
            continue
        
        # Handle root node: "Root" -> "True" (always true condition)
        if cond == "Root" or cond.strip() == "":
            python_conditions.append("True")
            continue
        
        # Handle missing value indicator: "(with NA)"
        if "(with NA)" in cond:
            # Extract the base condition
            base_cond = cond.replace(" (with NA)", "").strip()
            # Extract variable name (first word)
            var_name = base_cond.split()[0]
            # Recursively process base condition and add NaN check
            base_python = _label_to_python_condition(base_cond, column_names, collapse)
            # Add NaN check using pd.isna()
            python_cond = f"({base_python} | pd.isna({var_name}))"
            python_conditions.append(python_cond)
            continue
        
        # If no pattern matches, try to use as-is (might be a simple condition)
        python_conditions.append(f"({cond})")
    
    # Join all conditions with " & "
    return " & ".join(python_conditions)


def assign_to_leaves(
    rules: List[str],
    profile: pd.DataFrame,
    covariates: Optional[List[str]] = None
) -> np.ndarray:
    """
    Assign sequences to leaf nodes based on classification rules.
    
    Given a set of classification rules and a profile (covariate values),
    determines which leaf node each profile belongs to.
    
    **Corresponds to TraMineR function: `disstree.assign()`**
    
    Parameters
    ----------
    rules : List[str]
        List of classification rules (from get_classification_rules())
        
    profile : pd.DataFrame
        DataFrame with covariate values. Each row is a sequence/profile.
        
    covariates : List[str], optional
        List of covariate names to use. If None, uses all columns in profile.
        Default: None
        
    Returns
    -------
    np.ndarray
        Array of leaf node indices (1-based). Each element indicates which
        rule (leaf) applies to the corresponding row in profile.
        Returns NaN if no rule matches.
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import get_classification_rules, assign_to_leaves
    >>> 
    >>> # Get rules from tree
    >>> rules = get_classification_rules(tree)
    >>> 
    >>> # Create profile
    >>> profile = pd.DataFrame({'gender': ['M'], 'age': [25]})
    >>> 
    >>> # Assign to leaf
    >>> leaf_idx = assign_to_leaves(rules, profile)
    >>> print(f"Assigned to leaf: {leaf_idx[0]}")
    """
    if covariates is None:
        covariates = list(profile.columns)
    
    # Check all covariates are present
    missing = [c for c in covariates if c not in profile.columns]
    if missing:
        raise ValueError(
            f"[!] Missing covariates in profile: {missing}"
        )
    
    profile_subset = profile[covariates]
    n_profiles = len(profile_subset)
    assignments = np.full(n_profiles, np.nan, dtype=float)
    
    # Evaluate each rule for each profile
    for i, row in profile_subset.iterrows():
        # Create local namespace with covariate values
        local_vars = row.to_dict()
        
        # Add pandas functions for NaN handling
        import pandas as pd
        local_vars['pd'] = pd
        local_vars['np'] = np
        
        for rule_idx, rule in enumerate(rules):
            try:
                # Evaluate rule
                # Note: This requires rules to be valid Python expressions
                result = eval(rule, {"__builtins__": {}, "pd": pd, "np": np}, local_vars)
                if result:
                    assignments[i] = rule_idx + 1  # 1-based indexing
                    break
            except Exception as e:
                # Rule evaluation failed, skip
                continue
    
    return assignments
