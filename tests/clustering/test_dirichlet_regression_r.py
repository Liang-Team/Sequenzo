import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering.fuzzy_clustering.fuzzy_regression import (
    dirichlet_regression,
    prepare_dirichlet_data,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REFERENCE_JSON = FIXTURES / "dirichlet_r_reference.json"


def _load_reference() -> dict:
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def biofam_dirichlet_fixtures():
    membership = pd.read_csv(FIXTURES / "dirichlet_membership_raw.csv").to_numpy(dtype=np.float64)
    covariates = pd.read_csv(FIXTURES / "dirichlet_covariates.csv")
    fmember = pd.read_csv(FIXTURES / "dirichlet_fmember.csv").to_numpy(dtype=np.float64)
    reference = _load_reference()
    return membership, covariates, fmember, reference


def test_prepare_dirichlet_data_matches_r_dr_data(biofam_dirichlet_fixtures):
    membership, _, fmember, _ = biofam_dirichlet_fixtures
    prepared = prepare_dirichlet_data(membership)
    assert prepared.normalized is False
    assert prepared.transformed is False
    np.testing.assert_allclose(prepared.values, fmember, rtol=0.0, atol=1e-12)


def test_dirichlet_regression_matches_r_reference(biofam_dirichlet_fixtures):
    membership, covariates, _, reference = biofam_dirichlet_fixtures
    result = dirichlet_regression("sex + birthyr", covariates, membership, model="alternative")

    assert result.parametrization == "alternative"
    np.testing.assert_allclose(result.loglik, reference["loglik"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.aic, reference["aic"], rtol=0.0, atol=1e-4)

    for key in ("v2", "v3", "v4", "v5"):
        np.testing.assert_allclose(
            result.beta[key],
            np.asarray(reference["beta"][key], dtype=np.float64),
            rtol=0.0,
            atol=1e-5,
            err_msg=f"beta mismatch for {key}",
        )
    np.testing.assert_allclose(
        result.gamma["gamma"],
        np.asarray(reference["gamma"]["gamma"], dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    )
    assert result.beta["v1"] is None


def test_dirichlet_regression_returns_finite_loglik(biofam_dirichlet_fixtures):
    membership, covariates, _, _ = biofam_dirichlet_fixtures
    result = dirichlet_regression("sex + birthyr", covariates, membership)
    assert np.isfinite(result.loglik)
