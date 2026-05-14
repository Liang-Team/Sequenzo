import warnings

import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering.fuzzy_clustering.fuzzy_helpers import most_typical_members
from sequenzo.clustering.fuzzy_clustering.fuzzy_sequence_plots import _validate_membership


def test_validate_membership_warns_when_rows_do_not_sum_to_one():
    membership = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float)
    with pytest.warns(UserWarning, match="do not sum to 1"):
        out = _validate_membership(membership)
    np.testing.assert_array_equal(out, membership)


def test_validate_membership_does_not_warn_for_uniform_scaling_bug():
    """Rows summing to 2 should warn, not pass silently."""
    membership = np.full((3, 2), 1.0, dtype=float)
    with pytest.warns(UserWarning, match="do not sum to 1"):
        _validate_membership(membership)


def test_validate_membership_allows_possibilistic_partition():
    membership = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _validate_membership(membership, require_row_stochastic=False)
    np.testing.assert_array_equal(out, membership)


def test_most_typical_members_top_one():
    membership = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    result = most_typical_members(membership, labels=["a", "b", "c"])
    assert list(result.columns) == ["cluster", "rank", "index", "membership", "label"]
    assert result.loc[result["cluster"] == "1", "index"].iloc[0] == 0
    assert result.loc[result["cluster"] == "1", "membership"].iloc[0] == pytest.approx(0.9)
    assert result.loc[result["cluster"] == "2", "index"].iloc[0] == 1
    assert result.loc[result["cluster"] == "2", "membership"].iloc[0] == pytest.approx(0.8)


def test_most_typical_members_top_n():
    membership = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    result = most_typical_members(membership, top_n=2)
    cluster1 = result[result["cluster"] == "1"]
    assert list(cluster1["index"]) == [0, 1]
    assert list(cluster1["rank"]) == [1, 2]


def test_beta_regression_design_has_single_intercept():
    pytest.importorskip("statsmodels")
    from statsmodels.othermod.betareg import BetaModel
    import patsy

    from sequenzo.clustering.fuzzy_clustering.fuzzy_regression import beta_regression

    rng = np.random.default_rng(0)
    n = 30
    data = pd.DataFrame(
        {
            "sex": rng.integers(0, 2, size=n),
            "birthyr": rng.integers(1960, 1990, size=n),
        }
    )
    membership = rng.uniform(0.05, 0.95, size=n)
    reg_data = data.copy()
    reg_data["_membership_y"] = membership
    design = patsy.dmatrix("sex + birthyr", reg_data, return_type="dataframe")
    assert design.shape[1] == 3
    assert "Intercept" in design.columns

    result = beta_regression("sex + birthyr", data, membership)
    assert result.model.exog.shape[1] == 3
