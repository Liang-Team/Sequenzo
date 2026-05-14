"""
@Author  : Yuqi Liang 梁彧祺
@File    : _dirichlet_fit.py
@Time    : 14/05/2026 10:27
@Desc    :
Internal Dirichlet regression fitting (alternative parameterisation).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import patsy
import pandas as pd
from scipy.optimize import minimize_scalar, root
from scipy.special import gammaln, psi as digamma


def _starting_values_alternative(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    base_idx: int,
    weights: np.ndarray,
) -> np.ndarray:
    n_components = y.shape[1]
    n_mean_params = x.shape[1]
    non_base = [idx for idx in range(n_components) if idx != base_idx]

    y_log_ratio = np.log(y[:, non_base] / y[:, base_idx : base_idx + 1])
    x_weighted = x * weights[:, None]
    xtwx = x.T @ x_weighted
    beta_blocks = [
        np.linalg.solve(xtwx, x.T @ (weights * y_log_ratio[:, col]))
        for col in range(len(non_base))
    ]
    beta_mean = np.concatenate(beta_blocks)

    epsilon = np.ones((y.shape[0], n_components), dtype=np.float64)
    offset = 0
    for component_idx in non_base:
        epsilon[:, component_idx] = np.exp(x @ beta_mean[offset : offset + n_mean_params])
        offset += n_mean_params
    mu = epsilon / epsilon.sum(axis=1, keepdims=True)

    def neg_loglik_scalar(phi: float) -> float:
        alpha = mu * np.exp(phi)
        alpha0 = alpha.sum(axis=1)
        loglik = (
            gammaln(alpha0)
            - gammaln(alpha).sum(axis=1)
            + ((alpha - 1.0) * np.log(y)).sum(axis=1)
        )
        return -float(np.sum(weights * loglik))

    phi_hat = minimize_scalar(neg_loglik_scalar, bounds=(-20.0, 20.0), method="bounded").x
    gamma = np.linalg.lstsq(z, np.full(z.shape[0], phi_hat), rcond=None)[0]
    return np.concatenate([beta_mean, gamma])


def _unpack_theta(
    theta: np.ndarray,
    non_base: list[int],
    n_mean_params: int,
    n_precision_params: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    betas: dict[int, np.ndarray] = {}
    offset = 0
    for component_idx in non_base:
        betas[component_idx] = theta[offset : offset + n_mean_params]
        offset += n_mean_params
    gamma = theta[offset : offset + n_precision_params]
    return betas, gamma


def _stable_probabilities(
    betas: dict[int, np.ndarray],
    x: np.ndarray,
    base_idx: int,
    n_components: int,
) -> np.ndarray:
    non_base = [idx for idx in range(n_components) if idx != base_idx]
    etas = np.column_stack([x @ betas[idx] for idx in non_base])
    max_eta = np.maximum(0.0, np.max(etas, axis=1))
    log_denom = max_eta + np.log(np.exp(-max_eta) + np.sum(np.exp(etas - max_eta[:, None]), axis=1))
    probabilities = np.zeros((x.shape[0], n_components), dtype=np.float64)
    probabilities[:, base_idx] = np.exp(-log_denom)
    for column, component_idx in enumerate(non_base):
        probabilities[:, component_idx] = np.exp(etas[:, column] - log_denom)
    return probabilities, non_base, etas


def _mean_precision(
    betas: dict[int, np.ndarray],
    gamma: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    base_idx: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    probabilities, non_base, etas = _stable_probabilities(betas, x, base_idx, n_components)
    precision = np.exp(z @ gamma)[:, None]
    alpha = probabilities * precision
    return probabilities, precision, alpha, non_base, etas


def _neg_loglik_and_grad(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    base_idx: int,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    n_components = y.shape[1]
    n_mean_params = x.shape[1]
    n_precision_params = z.shape[1]
    non_base = [idx for idx in range(n_components) if idx != base_idx]
    betas, gamma = _unpack_theta(theta, non_base, n_mean_params, n_precision_params)
    probabilities, precision, alpha, non_base, _ = _mean_precision(
        betas, gamma, x, z, base_idx, n_components
    )

    alpha0 = alpha.sum(axis=1)
    log_y = np.log(y)
    score_alpha = digamma(alpha0)[:, None] - digamma(alpha) + log_y

    loglik = (
        gammaln(alpha0)
        - gammaln(alpha).sum(axis=1)
        + ((alpha - 1.0) * log_y).sum(axis=1)
    )
    neg_loglik = -float(np.sum(weights * loglik))
    grad = np.zeros_like(theta)

    precision_flat = precision[:, 0]
    gamma_coeff = (score_alpha * alpha).sum(axis=1)
    grad[(n_components - 1) * n_mean_params :] = -(z.T @ (weights * gamma_coeff))

    offset = 0
    for component_idx in non_base:
        pi_j = probabilities[:, component_idx]
        chain = np.zeros(x.shape[0], dtype=np.float64)
        for alt_idx in range(n_components):
            if alt_idx == component_idx:
                dpi_deta = pi_j * (1.0 - pi_j)
            elif alt_idx == base_idx:
                dpi_deta = -probabilities[:, base_idx] * pi_j
            else:
                dpi_deta = -probabilities[:, alt_idx] * pi_j
            chain += (score_alpha[:, alt_idx] * precision_flat * dpi_deta)
        grad[offset : offset + n_mean_params] = -(x.T @ (weights * chain))
        offset += n_mean_params

    return neg_loglik, grad


def fit_dirichlet_alternative(
    y: np.ndarray,
    mean_formula: str,
    precision_formula: str,
    data: pd.DataFrame,
    base: int = 1,
    weights: Optional[np.ndarray] = None,
) -> tuple[dict[str, Optional[np.ndarray]], dict[str, np.ndarray], float, int, list[str], list[str]]:
    if base < 1 or base > y.shape[1]:
        raise ValueError("base must be between 1 and the number of membership columns.")

    reg_data = data.copy()
    x = patsy.dmatrix(mean_formula, reg_data, return_type="dataframe")
    z = patsy.dmatrix(precision_formula, reg_data, return_type="dataframe")
    x_names = [str(col) for col in x.design_info.column_names]
    z_names = [str(col) for col in z.design_info.column_names]
    x_values = np.asarray(x, dtype=np.float64)
    z_values = np.asarray(z, dtype=np.float64)

    if weights is None:
        weights_arr = np.ones(y.shape[0], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.shape[0] != y.shape[0]:
            raise ValueError("weights must have one value per row.")

    base_idx = base - 1
    non_base = [idx for idx in range(y.shape[1]) if idx != base_idx]
    theta0 = _starting_values_alternative(y, x_values, z_values, base_idx, weights_arr)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return _neg_loglik_and_grad(theta, y, x_values, z_values, base_idx, weights_arr)[1]

    solution = root(gradient, theta0, method="hybr")
    if not solution.success:
        raise RuntimeError(f"Dirichlet regression optimisation failed: {solution.message}")

    betas_map, gamma = _unpack_theta(
        solution.x,
        non_base,
        x_values.shape[1],
        z_values.shape[1],
    )

    beta_out: dict[str, Optional[np.ndarray]] = {}
    for component_idx in range(y.shape[1]):
        key = f"v{component_idx + 1}"
        if component_idx == base_idx:
            beta_out[key] = None
        else:
            beta_out[key] = betas_map[component_idx]

    gamma_out = {"gamma": gamma}
    n_params = int(solution.x.size)
    loglik = -float(_neg_loglik_and_grad(solution.x, y, x_values, z_values, base_idx, weights_arr)[0])
    return beta_out, gamma_out, loglik, n_params, x_names, z_names
