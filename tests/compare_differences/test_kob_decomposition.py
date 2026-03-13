"""
Basic tests for the generic Kitagawa–Oaxaca–Blinder (KOB) decomposition.

These tests do not replicate a specific published application but ensure that:

- The total gap equals the difference in group means.
- The decomposition identity holds:
  total_gap ~= explained + unexplained_returns + unexplained_intercept.
- Variable-level contributions sum to the corresponding components.
"""

import numpy as np

from sequenzo.compare_differences import kob_decomposition


def test_kob_decomposition_identity_simple():
    """
    Simple sanity check on a small synthetic dataset.

    We construct a toy example where:
    - Group 0 has higher mean X than group 1.
    - Outcome y is a linear function of X plus different intercepts by group.

    The KOB decomposition should:
    - Recover the total mean gap between groups.
    - Satisfy the decomposition identity:
      total_gap ~= explained + unexplained_returns + unexplained_intercept.
    """
    rng = np.random.default_rng(123)

    n0, n1 = 50, 60
    # Covariate: group 0 has larger X on average
    X0 = rng.normal(loc=1.0, scale=0.5, size=(n0, 1))
    X1 = rng.normal(loc=0.0, scale=0.5, size=(n1, 1))

    # Outcome: y = alpha_g + beta * X + noise
    beta_true = 2.0
    alpha0_true = 5.0
    alpha1_true = 3.0

    y0 = alpha0_true + beta_true * X0[:, 0] + rng.normal(scale=0.1, size=n0)
    y1 = alpha1_true + beta_true * X1[:, 0] + rng.normal(scale=0.1, size=n1)

    y = np.concatenate([y0, y1])
    group = np.array([0] * n0 + [1] * n1)
    X = np.vstack([X0, X1])

    result = kob_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=["X1"],
        term_ids=[0],
        reference="group0",
    )

    # Total gap should match mean difference
    gap_direct = y[group == 0].mean() - y[group == 1].mean()
    assert np.isclose(result.total_gap, gap_direct, atol=1e-6)

    # Decomposition identity
    recon_gap = result.explained + result.unexplained_returns + result.unexplained_intercept
    assert np.isclose(result.total_gap, recon_gap, atol=1e-6)

    # Variable-level contributions sum up correctly
    explained_sum = result.by_variable["explained"].sum()
    returns_sum = result.by_variable["returns"].sum()

    assert np.isclose(explained_sum, result.explained, atol=1e-6)
    assert np.isclose(returns_sum, result.unexplained_returns, atol=1e-6)

