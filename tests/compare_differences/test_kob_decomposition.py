"""
Basic tests for the generic Kitagawa–Oaxaca–Blinder (KOB) decomposition.
"""

import warnings

import numpy as np
import pytest

from sequenzo.group_comparison import get_kob_decomposition
from sequenzo.decomposition import get_kob_decomposition_bootstrap


def _assert_decomposition_identity(
    result,
    *,
    atol=1e-6,
    check_column_sums: bool = True,
    check_term_sums: bool = True,
):
    recon_gap = result.explained + result.unexplained_returns + result.unexplained_intercept
    assert np.isclose(result.total_gap, recon_gap, atol=atol)
    if check_column_sums:
        assert np.isclose(result.by_column["explained"].sum(), result.explained, atol=atol)
        assert np.isclose(result.by_column["returns"].sum(), result.unexplained_returns, atol=atol)
    if check_term_sums:
        assert np.isclose(result.by_term["explained"].sum(), result.explained, atol=atol)
        assert np.isclose(result.by_term["returns"].sum(), result.unexplained_returns, atol=atol)


def test_kob_decomposition_identity_simple():
    rng = np.random.default_rng(123)

    n0, n1 = 50, 60
    X0 = rng.normal(loc=1.0, scale=0.5, size=(n0, 1))
    X1 = rng.normal(loc=0.0, scale=0.5, size=(n1, 1))

    beta_true = 2.0
    alpha0_true = 5.0
    alpha1_true = 3.0

    y0 = alpha0_true + beta_true * X0[:, 0] + rng.normal(scale=0.1, size=n0)
    y1 = alpha1_true + beta_true * X1[:, 0] + rng.normal(scale=0.1, size=n1)

    y = np.concatenate([y0, y1])
    group = np.array([0] * n0 + [1] * n1)
    X = np.vstack([X0, X1])

    result = get_kob_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=["X1"],
        term_ids=[0],
        reference="group0",
        group0_value=0,
        group1_value=1,
    )

    gap_direct = y[group == 0].mean() - y[group == 1].mean()
    assert np.isclose(result.total_gap, gap_direct, atol=1e-6)
    assert result.group0_label == 0
    assert result.group1_label == 1
    assert result.gap_direction == "0 minus 1"
    _assert_decomposition_identity(result)


def test_group_order_is_explicit_for_string_labels():
    rng = np.random.default_rng(7)
    n = 40
    X = rng.normal(size=(n, 1))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=n)
    group = np.array(["men"] * (n // 2) + ["women"] * (n - n // 2))

    men_first = get_kob_decomposition(
        y=y,
        group=group,
        X=X,
        group0_value="men",
        group1_value="women",
    )
    women_first = get_kob_decomposition(
        y=y,
        group=group,
        X=X,
        group0_value="women",
        group1_value="men",
    )

    assert np.isclose(men_first.total_gap, -women_first.total_gap, atol=1e-10)


def test_coefficient_owner_by_column_cluster_specific_reference():
    rng = np.random.default_rng(11)
    n0, n1 = 80, 80
    cluster_props0 = np.array([0.5, 0.3, 0.2])
    cluster_props1 = np.array([0.2, 0.3, 0.5])
    X0 = rng.multinomial(1, cluster_props0, size=n0)[:, :2].astype(float)
    X1 = rng.multinomial(1, cluster_props1, size=n1)[:, :2].astype(float)
    beta = np.array([1.0, 2.0])
    y0 = X0 @ beta + rng.normal(scale=0.1, size=n0)
    y1 = X1 @ beta + 0.5 + rng.normal(scale=0.1, size=n1)

    y = np.concatenate([y0, y1])
    group = np.array([0] * n0 + [1] * n1)
    X = np.vstack([X0, X1])

    result = get_kob_decomposition(
        y=y,
        group=group,
        X=X,
        term_ids=[0, 0],
        coefficient_owner_by_column=[0, 1],
    )
    _assert_decomposition_identity(result)


def test_categorical_normalization_is_reference_invariant():
    n0, n1 = 120, 120
    probs0 = np.array([0.4, 0.35, 0.25])
    probs1 = np.array([0.25, 0.35, 0.4])
    rng = np.random.default_rng(21)
    draws0 = rng.multinomial(1, probs0, size=n0)
    draws1 = rng.multinomial(1, probs1, size=n1)
    beta_true = np.array([1.0, 1.5, 2.0])
    y0 = draws0 @ beta_true + rng.normal(scale=0.1, size=n0)
    y1 = draws1 @ beta_true + 0.25 + rng.normal(scale=0.1, size=n1)
    y = np.concatenate([y0, y1])
    group = np.array([0] * n0 + [1] * n1)

    def _decompose(drop_idx: int):
        keep = [i for i in range(3) if i != drop_idx]
        X = np.vstack([draws0[:, keep], draws1[:, keep]]).astype(float)
        return get_kob_decomposition(
            y=y,
            group=group,
            X=X,
            term_ids=[0, 0],
            category_ids=keep,
            normalize_categorical=True,
            categorical_terms=[0],
            n_categories_by_term={0: 3},
        )

    res_drop0 = _decompose(drop_idx=0)
    res_drop1 = _decompose(drop_idx=1)
    res_drop2 = _decompose(drop_idx=2)

    assert np.allclose(res_drop0.by_term["explained"].to_numpy(), res_drop1.by_term["explained"].to_numpy(), atol=1e-6)
    assert np.allclose(res_drop0.by_term["returns"].to_numpy(), res_drop1.by_term["returns"].to_numpy(), atol=1e-6)
    assert np.allclose(res_drop0.by_term["explained"].to_numpy(), res_drop2.by_term["explained"].to_numpy(), atol=1e-6)
    assert np.allclose(
        res_drop0.by_category["explained"].to_numpy(),
        res_drop1.by_category["explained"].to_numpy(),
        atol=1e-6,
    )
    _assert_decomposition_identity(res_drop0, check_column_sums=False, check_term_sums=False)


def test_invalid_majority_owner_raises():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    group = np.array([0, 0, 1, 1])
    X = np.array([[1.0], [2.0], [3.0], [4.0]])

    with pytest.raises(ValueError, match="majority_owner must contain only"):
        get_kob_decomposition(y=y, group=group, X=X, majority_owner=[2])


def test_nan_input_raises():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    group = np.array([0, 0, 1, 1])
    X = np.array([[1.0], [2.0], [3.0], [4.0]])

    with pytest.raises(ValueError, match="NaN or infinite"):
        get_kob_decomposition(y=y, group=group, X=X)


def test_bootstrap_runs_and_returns_uncertainty():
    rng = np.random.default_rng(99)
    n0, n1 = 40, 45
    X0 = rng.normal(loc=1.0, scale=0.5, size=(n0, 1))
    X1 = rng.normal(loc=0.0, scale=0.5, size=(n1, 1))
    y0 = 5.0 + 2.0 * X0[:, 0] + rng.normal(scale=0.1, size=n0)
    y1 = 3.0 + 2.0 * X1[:, 0] + rng.normal(scale=0.1, size=n1)
    y = np.concatenate([y0, y1])
    group = np.array([0] * n0 + [1] * n1)
    X = np.vstack([X0, X1])

    boot = get_kob_decomposition_bootstrap(
        y=y,
        group=group,
        X=X,
        n_boot=50,
        random_state=1,
    )

    assert boot.n_boot == 50
    assert boot.standard_errors["total_gap"] >= 0.0
    assert boot.confidence_intervals["total_gap"][0] <= boot.confidence_intervals["total_gap"][1]
    _assert_decomposition_identity(boot.point_estimate)


def test_rank_deficient_design_warns():
    rng = np.random.default_rng(3)
    n = 30
    x = rng.normal(size=n)
    X = np.column_stack([x, x])
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    group = np.array([0] * (n // 2) + [1] * (n - n // 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_kob_decomposition(y=y, group=group, X=X)
        assert any("rank deficient" in str(w.message).lower() for w in caught)
    assert result.diagnostics["group0"]["ols"]["rank_deficient"] is True
