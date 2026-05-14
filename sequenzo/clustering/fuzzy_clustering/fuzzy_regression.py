"""
@Author  : Yuqi Liang 梁彧祺
@File    : fuzzy_regression.py
@Time    : 14/05/2026
@Desc    :
Regression helpers for fuzzy membership matrices (Studer 2018 R tutorial).

Mirrors ``DirichletReg::DirichReg`` and ``betareg::betareg`` workflows for
analysing fuzzy cluster membership as a dependent variable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from ._dirichlet_fit import fit_dirichlet_alternative


@dataclass
class DirichletRegData:
    """Python mirror of R ``DirichletRegData``."""

    values: np.ndarray
    normalized: bool = False
    transformed: bool = False
    base: int = 1
    column_names: list[str] = field(default_factory=list)


@dataclass
class DirichletRegResult:
    """Structured result from :func:`dirichlet_regression`."""

    beta: dict[str, Optional[np.ndarray]]
    gamma: dict[str, np.ndarray]
    loglik: float
    aic: float
    parametrization: str
    base: int
    beta_names: dict[str, Optional[list[str]]]
    gamma_names: dict[str, list[str]]
    n_params: int
    prepared_data: Optional[DirichletRegData] = None


def prepare_dirichlet_data(
    membership: Union[np.ndarray, pd.DataFrame],
    trafo: Union[bool, float] = math.sqrt(np.finfo(float).eps),
    base: int = 1,
    norm_tol: Optional[float] = None,
) -> DirichletRegData:
    """
    Format a membership matrix for Dirichlet regression.

    Mirrors ``DirichletReg::DR_data`` (Maier 2014). Rows should be non-negative.
    Rows are renormalized when they do not sum to 1 within ``norm_tol``. Boundary
    values (0 or 1) trigger the same interior transform as R when ``trafo`` is
    logical ``True`` or a small numeric threshold.
    """
    if norm_tol is None:
        norm_tol = math.sqrt(np.finfo(float).eps)

    if isinstance(membership, pd.DataFrame):
        values = membership.to_numpy(dtype=np.float64, copy=True)
        column_names = [str(col) for col in membership.columns]
    else:
        values = np.asarray(membership, dtype=np.float64).copy()
        column_names = []

    if values.ndim != 2:
        raise ValueError("membership must be a 2D matrix.")
    if values.shape[1] <= 1:
        raise ValueError("membership must have at least two columns.")
    if base < 1 or base > values.shape[1]:
        raise ValueError("base must be between 1 and the number of membership columns.")
    if np.any(values < 0):
        raise ValueError("membership values must be non-negative.")

    n_components = values.shape[1]
    if not column_names:
        column_names = [f"v{idx}" for idx in range(1, n_components + 1)]

    force_norm = False
    row_sums = values.sum(axis=1)
    positive_rows = row_sums[row_sums > 0]
    if positive_rows.size and not np.allclose(positive_rows, 1.0, rtol=0.0, atol=norm_tol):
        values = values / row_sums[:, None]
        force_norm = True

    force_tran = False
    finite = values[np.isfinite(values)]
    needs_trafo = False
    if isinstance(trafo, bool):
        needs_trafo = trafo
        force_tran = False
    elif isinstance(trafo, (int, float)):
        if trafo < 0 or trafo >= 0.5:
            raise ValueError("numeric trafo must be > 0 and < 0.5.")
        needs_trafo = np.any(finite < trafo) or np.any(finite > (1.0 - trafo))
        force_tran = needs_trafo
    else:
        raise ValueError("trafo must be a bool or a small positive float.")

    if needs_trafo:
        n_obs = int(np.sum(np.isfinite(row_sums)))
        values = (values * (n_obs - 1) + 1.0 / n_components) / n_obs

    if np.any(values <= 0) or np.any(values >= 1):
        raise ValueError(
            "membership still contains exact 0 or 1 after preparation; "
            "consider enabling trafo."
        )

    return DirichletRegData(
        values=values,
        normalized=bool(force_norm),
        transformed=bool(force_tran),
        base=base,
        column_names=column_names,
    )


def dirichlet_regression(
    formula: str,
    data: pd.DataFrame,
    membership: Union[np.ndarray, pd.DataFrame],
    model: Literal["alternative", "common"] = "alternative",
    base: int = 1,
    precision_formula: str = "1",
    weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> DirichletRegResult:
    """
    Fit a Dirichlet regression model to a fuzzy membership matrix.

    Mirrors ``DirichReg(fmember ~ covariates | 1, data=..., model="alternative")``
    from the Studer (2018) / WeightedCluster ``FuzzySA`` tutorial.

    This is a pure Python implementation of the R ``DirichletReg`` alternative
    parameterisation (Maier 2014): multinomial-logit means on the simplex with
    a log-linear precision term. Estimates are obtained by solving the score
    equations with the same starting-value strategy as R.

    Parameters
    ----------
    formula : str
        Right-hand side of the mean model, e.g. ``"sex + birthyr"``.
    data : pd.DataFrame
        Covariate data with one row per sequence.
    membership : array-like, shape (n, k)
        Fuzzy membership matrix. Prepared internally via :func:`prepare_dirichlet_data`.
    model : {"alternative", "common"}, default "alternative"
        Parameterisation. Only ``"alternative"`` is currently supported.
    base : int, default 1
        Reference category for the alternative parameterisation (1-based, as in R).
    precision_formula : str, default "1"
        Right-hand side of the precision sub-model after ``|``.
    weights : np.ndarray, optional
        Observation weights.
    **kwargs
        Reserved for future extensions.

    Returns
    -------
    DirichletRegResult
    """
    del kwargs
    if model != "alternative":
        raise NotImplementedError('Only model="alternative" is currently supported.')

    if isinstance(membership, pd.DataFrame):
        membership_values = membership.to_numpy(dtype=np.float64)
    else:
        membership_values = np.asarray(membership, dtype=np.float64)

    if len(data) != membership_values.shape[0]:
        raise ValueError("data must have one row per membership row.")

    prepared = prepare_dirichlet_data(membership_values, base=base)
    beta, gamma, loglik, n_params, x_names, z_names = fit_dirichlet_alternative(
        prepared.values,
        mean_formula=formula,
        precision_formula=precision_formula,
        data=data,
        base=base,
        weights=weights,
    )

    beta_names: dict[str, Optional[list[str]]] = {}
    for key, values in beta.items():
        beta_names[key] = None if values is None else list(x_names)

    gamma_names = {"gamma": list(z_names)}
    aic = -2.0 * loglik + 2.0 * n_params

    return DirichletRegResult(
        beta=beta,
        gamma=gamma,
        loglik=loglik,
        aic=aic,
        parametrization="alternative",
        base=base,
        beta_names=beta_names,
        gamma_names=gamma_names,
        n_params=n_params,
        prepared_data=prepared,
    )


def beta_regression(
    formula: str,
    data: pd.DataFrame,
    membership: Union[np.ndarray, pd.Series, pd.DataFrame],
    membership_column: Optional[Union[int, str]] = None,
    **kwargs: Any,
):
    """
    Fit a beta regression to one column of a fuzzy membership matrix.

    Mirrors ``betareg(fclust$membership[, j] ~ covariates, data=...)``.

    Uses ``statsmodels.othermod.betareg.BetaModel`` when available.

    Parameters
    ----------
    formula : str
        Model formula without the dependent variable, e.g. ``"sex + birthyr"``.
    data : pd.DataFrame
        Covariate data with one row per sequence.
    membership : array-like
        Full membership matrix or a single membership vector.
    membership_column : int or str, optional
        Column index (0-based) or name when ``membership`` is a matrix.
        Required unless ``membership`` is already one-dimensional.

    Returns
    -------
    statsmodels results object
    """
    try:
        from statsmodels.othermod.betareg import BetaModel
        import patsy
    except ImportError as exc:
        raise ImportError(
            "beta_regression requires statsmodels with BetaModel support "
            "(statsmodels >= 0.12)."
        ) from exc

    if isinstance(membership, pd.DataFrame):
        if membership_column is None:
            raise ValueError("membership_column is required for a DataFrame membership matrix.")
        y = membership[membership_column].to_numpy(dtype=np.float64)
    elif isinstance(membership, pd.Series):
        y = membership.to_numpy(dtype=np.float64)
    else:
        membership = np.asarray(membership, dtype=np.float64)
        if membership.ndim == 2:
            if membership_column is None:
                raise ValueError("membership_column is required for a 2D membership matrix.")
            col = int(membership_column)
            y = membership[:, col]
        else:
            y = membership.reshape(-1)

    eps = 1e-6
    y = np.clip(y, eps, 1.0 - eps)
    reg_data = data.copy()
    reg_data["_membership_y"] = y
    design = patsy.dmatrix(formula, reg_data, return_type="dataframe")
    return BetaModel(y, design).fit(**kwargs)
