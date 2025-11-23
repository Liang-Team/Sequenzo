"""
@Author  : Yuqi Liang 梁彧祺
@File    : formulas.py
@Time    : 2025-10-18 16:23
@Desc    : Formula-based covariate specification for NHMM

This module provides a formula interface for specifying covariates in NHMM,
similar to seqHMM's formula interface in R. Users can specify covariates
using a string formula like "~ x1 + x2" instead of manually creating
covariate matrices.

Note: This is a simplified implementation. A full implementation would
support more complex formulas (interactions, transformations, etc.).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict
from sequenzo.define_sequence_data import SequenceData


class Formula:
    """
    Formula object for specifying covariates.
    
    This class represents a formula like "~ x1 + x2" and can be used
    to create model matrices from data.
    
    Examples:
        >>> formula = Formula("~ age + gender")
        >>> X = formula.create_matrix(data, id_var='id', time_var='time')
    """
    
    def __init__(self, formula: str):
        """
        Initialize a formula object.
        
        Args:
            formula: Formula string, e.g., "~ x1 + x2" or "x1 + x2"
                    (tilde is optional)
        """
        # Remove leading/trailing whitespace
        formula = formula.strip()
        
        # Remove tilde if present
        if formula.startswith('~'):
            formula = formula[1:].strip()
        
        self.formula = formula
        self.terms = self._parse_formula(formula)
    
    def _parse_formula(self, formula: str) -> List[str]:
        """
        Parse formula string into terms.
        
        Args:
            formula: Formula string
            
        Returns:
            List of variable names
        """
        if not formula:
            return []
        
        # Split by + and clean up
        terms = [term.strip() for term in formula.split('+')]
        return [t for t in terms if t]  # Remove empty strings
    
    def create_matrix(
        self,
        data: pd.DataFrame,
        id_var: str,
        time_var: str,
        n_sequences: int,
        n_timepoints: int
    ) -> np.ndarray:
        """
        Create covariate matrix from formula and data.
        
        This function creates a covariate matrix X of shape
        (n_sequences, n_timepoints, n_covariates) from a DataFrame
        and formula specification.
        
        Args:
            data: DataFrame containing covariates
            id_var: Column name for sequence IDs
            time_var: Column name for time variable
            n_sequences: Number of sequences
            n_timepoints: Number of time points
            
        Returns:
            numpy array: Covariate matrix (n_sequences, n_timepoints, n_covariates)
        """
        if not self.terms:
            # No covariates: return matrix of ones (intercept only)
            return np.ones((n_sequences, n_timepoints, 1))
        
        # Initialize covariate matrix
        X = np.zeros((n_sequences, n_timepoints, len(self.terms) + 1))  # +1 for intercept
        
        # First column is intercept (always 1)
        X[:, :, 0] = 1.0
        
        # Fill in covariates
        for term_idx, term in enumerate(self.terms):
            col_idx = term_idx + 1  # +1 because first column is intercept
            
            if term not in data.columns:
                raise ValueError(f"Variable '{term}' not found in data columns: {list(data.columns)}")
            
            # Get values for this covariate
            covar_values = data[term].values
            
            # Reshape to match sequence structure
            # This assumes data is in long format (one row per sequence-time combination)
            # We need to reshape it to (n_sequences, n_timepoints)
            
            # If data has id_var and time_var, use them to reshape
            if id_var in data.columns and time_var in data.columns:
                # Pivot to wide format
                pivot_df = data.pivot(index=id_var, columns=time_var, values=term)
                
                # Fill matrix
                for seq_idx, seq_id in enumerate(pivot_df.index):
                    if seq_idx < n_sequences:
                        for t_idx, time_val in enumerate(pivot_df.columns):
                            if t_idx < n_timepoints:
                                X[seq_idx, t_idx, col_idx] = pivot_df.loc[seq_id, time_val]
            else:
                # Assume data is already in sequence-time order
                # Reshape assuming row-major order (sequence by sequence)
                if len(covar_values) == n_sequences * n_timepoints:
                    X[:, :, col_idx] = covar_values.reshape(n_sequences, n_timepoints)
                else:
                    raise ValueError(
                        f"Data length ({len(covar_values)}) doesn't match "
                        f"n_sequences * n_timepoints ({n_sequences * n_timepoints})"
                    )
        
        return X


def create_model_matrix(
    formula: Union[str, Formula],
    data: pd.DataFrame,
    id_var: str,
    time_var: str,
    n_sequences: int,
    n_timepoints: int
) -> np.ndarray:
    """
    Create model matrix from formula and data.
    
    This is a convenience function that creates a covariate matrix
    from a formula string, similar to seqHMM's model_matrix() function.
    
    Args:
        formula: Formula string (e.g., "~ x1 + x2") or Formula object
        data: DataFrame containing covariates
        id_var: Column name for sequence IDs
        time_var: Column name for time variable
        n_sequences: Number of sequences
        n_timepoints: Number of time points
        
    Returns:
        numpy array: Covariate matrix (n_sequences, n_timepoints, n_covariates)
        
    Examples:
        >>> import pandas as pd
        >>> from sequenzo.seqhmm import create_model_matrix
        >>> 
        >>> # Create data with covariates
        >>> data = pd.DataFrame({
        ...     'id': [1, 1, 1, 2, 2, 2],
        ...     'time': [1, 2, 3, 1, 2, 3],
        ...     'age': [20, 21, 22, 25, 26, 27],
        ...     'gender': [0, 0, 0, 1, 1, 1]
        ... })
        >>> 
        >>> # Create model matrix
        >>> X = create_model_matrix("~ age + gender", data, 'id', 'time', n_sequences=2, n_timepoints=3)
        >>> print(X.shape)  # (2, 3, 3) - 2 sequences, 3 timepoints, 3 covariates (intercept + age + gender)
    """
    if isinstance(formula, str):
        formula = Formula(formula)
    
    return formula.create_matrix(data, id_var, time_var, n_sequences, n_timepoints)


def create_model_matrix_time_constant(
    formula: Union[str, Formula, None],
    data: Optional[pd.DataFrame],
    n_sequences: int
) -> np.ndarray:
    """
    Create model matrix for time-constant covariates (one value per sequence).
    
    This function creates a model matrix for time-constant covariates used in
    MHMM simulation. The covariates are constant across time points for each sequence,
    so the output matrix has shape (n_sequences, n_covariates) where n_covariates
    includes an intercept column.
    
    This is similar to R's model.matrix() function but for time-constant covariates.
    
    Args:
        formula: Formula string (e.g., "~ covariate_1 + covariate_2") or Formula object.
                If None, returns a matrix with only intercept (column of ones).
        data: DataFrame containing covariates. Must have n_sequences rows.
               Each row corresponds to one sequence.
        n_sequences: Number of sequences to simulate
        
    Returns:
        numpy array: Model matrix of shape (n_sequences, n_covariates)
                    First column is always intercept (ones)
                    Subsequent columns are the covariates specified in formula
                    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sequenzo.seqhmm.formulas import create_model_matrix_time_constant
        >>> 
        >>> # Create covariate data (one row per sequence)
        >>> data = pd.DataFrame({
        ...     'covariate_1': np.random.rand(10),
        ...     'covariate_2': np.random.choice(['A', 'B'], size=10)
        ... })
        >>> 
        >>> # Create model matrix with formula
        >>> X = create_model_matrix_time_constant("~ covariate_1 + covariate_2", data, n_sequences=10)
        >>> print(X.shape)  # (10, n_covariates) where n_covariates includes intercept and dummies
    """
    # If no formula is provided, return intercept-only matrix
    if formula is None:
        return np.ones((n_sequences, 1))
    
    # Parse formula
    if isinstance(formula, str):
        formula = Formula(formula)
    
    # Validate data
    if data is None:
        raise ValueError("If formula is provided, data must also be provided")
    
    if len(data) != n_sequences:
        raise ValueError(
            f"Number of rows in data ({len(data)}) must equal n_sequences ({n_sequences})"
        )
    
    # Get terms from formula
    terms = formula.terms
    
    # Initialize model matrix with intercept column
    # We'll build it step by step, handling factor variables
    columns_list = []
    column_names = ['(Intercept)']
    
    # Add intercept column (all ones)
    columns_list.append(np.ones(n_sequences))
    
    # Process each term in the formula
    for term in terms:
        if term not in data.columns:
            raise ValueError(
                f"Variable '{term}' not found in data columns: {list(data.columns)}"
            )
        
        covar_values = data[term].values
        
        # Check if this is a categorical variable
        if pd.api.types.is_categorical_dtype(data[term]) or \
           pd.api.types.is_object_dtype(data[term]) or \
           (data[term].dtype == 'object'):
            # Categorical variable: create dummy variables
            # Use pandas get_dummies to create dummies, drop first level as reference
            dummies = pd.get_dummies(data[[term]], prefix=term, drop_first=True)
            
            # Add each dummy column
            for dummy_col in dummies.columns:
                columns_list.append(dummies[dummy_col].values)
                column_names.append(dummy_col)
        else:
            # Numeric variable: add as is
            columns_list.append(covar_values)
            column_names.append(term)
    
    # Stack all columns into a matrix
    X = np.column_stack(columns_list)
    
    return X
