"""
@Author  : Yuqi Liang 梁彧祺
@File    : formulas.py
@Time    : 2025-10-18 16:23
@Desc    : Formula-based covariate specification for NHMM

This module provides a formula interface for specifying covariates in NHMM,
similar to seqHMM's formula interface in R. Users can specify covariates
using string formulas, including patsy-supported interactions and transforms,
instead of manually creating covariate matrices.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Sequence

try:
    from patsy import EvalEnvironment, PatsyError, dmatrix
except ImportError:  # pragma: no cover - patsy is provided by statsmodels in normal installs
    EvalEnvironment = None
    PatsyError = None
    dmatrix = None


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
        self.raw_formula = formula
        self.lhs = None
        
        # Keep only the RHS when callers use an R-style ``response ~ terms`` formula.
        if "~" in formula:
            lhs, formula = formula.split("~", 1)
            self.lhs = lhs.strip() or None
            formula = formula.strip()
        
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
        if not formula or formula == '1':
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
        n_timepoints: int,
        id_values: Optional[Sequence] = None,
        time_values: Optional[Sequence] = None,
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
        if dmatrix is None:
            raise ImportError("patsy is required for formula model matrices")

        if id_var not in data.columns or time_var not in data.columns:
            raise ValueError(f"data must contain id_var={id_var!r} and time_var={time_var!r}")

        if data.duplicated([id_var, time_var]).any():
            raise ValueError("data must not contain duplicate id/time cells")

        if id_values is None:
            ids = list(pd.unique(data[id_var]))[:n_sequences]
        else:
            ids = list(id_values)
            if len(ids) != n_sequences:
                raise ValueError("id_values length must equal n_sequences")

        if time_values is None:
            times = list(pd.unique(data[time_var]))[:n_timepoints]
        else:
            times = list(time_values)
            if len(times) != n_timepoints:
                raise ValueError("time_values length must equal n_timepoints")

        if len(ids) < n_sequences or len(times) < n_timepoints:
            raise ValueError("data does not contain enough ids or time points")

        grid = pd.MultiIndex.from_product(
            [ids, times],
            names=[id_var, time_var],
        ).to_frame(index=False)
        grid_pairs = set(map(tuple, grid[[id_var, time_var]].to_numpy()))
        data_pairs = set(map(tuple, data[[id_var, time_var]].to_numpy()))
        if not grid_pairs.issubset(data_pairs):
            raise ValueError(
                "data must contain a complete id x time grid matching "
                "the SequenceData id and time order for formula covariates"
            )
        if id_values is not None and time_values is not None and data_pairs != grid_pairs:
            raise ValueError(
                "data must not contain id/time cells outside the SequenceData grid"
            )

        design_data = grid.merge(data, on=[id_var, time_var], how="left", validate="one_to_one")

        def lag(x, n=1, fill=0.0):
            n = int(n)
            values = pd.Series(x)
            numeric_values = pd.to_numeric(values, errors="coerce")
            if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
                return values.to_numpy()
            shifted = values.groupby(design_data[id_var], sort=False).shift(n)
            return shifted.fillna(fill).to_numpy()

        formula = _as_patsy_formula(self)
        env = EvalEnvironment.capture(0).with_outer_namespace({"np": np, "lag": lag})
        try:
            design = dmatrix(
                formula,
                design_data,
                return_type="dataframe",
                eval_env=env,
                NA_action="raise",
            )
        except PatsyError as exc:
            raise ValueError(
                "formula covariates must be non-missing and finite"
            ) from exc
        design = design.astype(float)
        if len(design) != len(design_data):
            raise ValueError(
                "formula covariates must preserve one row per id/time cell"
            )

        values = design.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("formula covariates must be non-missing and finite")
        return values.reshape(n_sequences, n_timepoints, design.shape[1])


def _as_patsy_formula(formula: Union[str, Formula, None]) -> str:
    if formula is None:
        return "~ 1"
    if isinstance(formula, Formula):
        rhs = formula.formula
    else:
        rhs = str(formula).strip()
        if rhs.startswith('~'):
            return rhs
    if not rhs:
        rhs = "1"
    return f"~ {rhs}"


def create_model_matrix(
    formula: Union[str, Formula],
    data: pd.DataFrame,
    id_var: str,
    time_var: str,
    n_sequences: int,
    n_timepoints: int,
    id_values: Optional[Sequence] = None,
    time_values: Optional[Sequence] = None,
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

    return formula.create_matrix(
        data,
        id_var,
        time_var,
        n_sequences,
        n_timepoints,
        id_values=id_values,
        time_values=time_values,
    )


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
    
    if dmatrix is None:
        raise ImportError("patsy is required for formula model matrices")

    formula_obj = formula
    formula_string = _as_patsy_formula(formula_obj)
    env = EvalEnvironment.capture(0).with_outer_namespace({"np": np})
    try:
        design = dmatrix(
            formula_string,
            data,
            return_type="dataframe",
            eval_env=env,
            NA_action="raise",
        )
    except PatsyError as exc:
        raise ValueError(
            "formula covariates must be non-missing and finite"
        ) from exc
    values = design.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("formula covariates must be non-missing and finite")

    return values
