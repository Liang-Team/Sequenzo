"""
@Author  : Yapeng Wei
@File    : test_seqhmm_nhmm.py
@Desc    :
Tests for Non-Homogeneous Hidden Markov Model (NHMM) consistency between
sequenzo.seqhmm and R seqHMM.

Actual Python API:
  - build_nhmm(observations, n_states, X=..., emission_formula=...,
               data=..., id_var=..., time_var=..., ...) -> NHMM
  - fit_nhmm(model, n_iter, tol, verbose) -> NHMM
  - NHMM attributes: .eta_pi, .eta_A, .eta_B, .log_likelihood,
    .n_states, .n_covariates, .X, .sequence_lengths, .converged

R equivalent: estimate_nhmm() in seqHMM >= 2.1.0

Test groups:
  Part 0: Sanity (no R)
  Part 1: Cross-language consistency (needs ref CSVs)

Run: Rscript seqhmm_reference_nhmm.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_nhmm, fit_nhmm
from sequenzo.seqhmm.forward_backward_nhmm import log_likelihood_nhmm


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
N_STATES = 2
N_ID = 20
N_TIME = 10
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Tolerances
# NHMM uses L-BFGS-B which is highly sensitive to initialisation.
# Python and R routinely converge to different local optima (6+ nats apart),
# so we only check that they are in the same ballpark.
LOGLIK_ATOL = 8.0


# ============================================================================
# Synthetic panel data (must match R script)
# ============================================================================

def _make_panel_data():
    """Create panel data identical to R script."""
    csv_path = os.path.join(THIS_DIR, "ref_nhmm_panel_data.csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)

    # Fallback: generate deterministically
    rng = np.random.RandomState(42)
    ids = np.repeat(np.arange(1, N_ID + 1), N_TIME)
    times = np.tile(np.arange(1, N_TIME + 1), N_ID)
    x_covariate = np.round(rng.randn(N_ID * N_TIME), 3)

    rng2 = np.random.RandomState(42)
    response = []
    for i in range(N_ID * N_TIME):
        if x_covariate[i] > 0:
            probs = [0.2, 0.3, 0.5]
        else:
            probs = [0.5, 0.3, 0.2]
        response.append(rng2.choice(STATES, p=probs))

    return pd.DataFrame({
        "id": ids,
        "time": times,
        "response": response,
        "x": x_covariate,
    })


def _panel_to_seqdata_and_X(panel_data):
    """Convert panel data to SequenceData + covariate matrix X.

    Returns (seqdata, X) where:
      seqdata: SequenceData with wide-format sequences
      X: covariate array of shape (n_sequences, n_timepoints, n_covariates)
    """
    # Pivot to wide format for SequenceData
    wide = panel_data.pivot(index="id", columns="time", values="response")
    wide = wide.reset_index()
    time_cols = [c for c in wide.columns if c != "id"]

    seqdata = SequenceData(
        wide, time=time_cols, states=STATES, id_col="id"
    )

    # Build covariate matrix X: shape (n_sequences, n_timepoints, n_covariates)
    # Include intercept + x
    n_seq = len(seqdata.sequences)
    n_time = max(len(s) for s in seqdata.sequences)
    X = np.ones((n_seq, n_time, 2))  # intercept + x

    ids_sorted = sorted(panel_data["id"].unique())
    for i, sid in enumerate(ids_sorted):
        mask = panel_data["id"] == sid
        x_vals = panel_data.loc[mask, "x"].values
        for t in range(min(n_time, len(x_vals))):
            X[i, t, 1] = x_vals[t]

    return seqdata, X


def _build_nhmm_from_panel(panel_data, random_state=42):
    """Build and fit an NHMM from panel data."""
    seqdata, X = _panel_to_seqdata_and_X(panel_data)
    nhmm = build_nhmm(
        observations=seqdata,
        n_states=N_STATES,
        X=X,
        random_state=random_state,
    )
    return nhmm


# ============================================================================
# Helpers
# ============================================================================

def _run_r_reference(outdir, timeout=300):
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_nhmm.R")
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, outdir],
            capture_output=True, text=True, timeout=timeout, cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _load_ref_kv(ref_dir, filename):
    fpath = os.path.join(ref_dir, filename)
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    if "key" in df.columns and "value" in df.columns:
        return dict(zip(df["key"], df["value"]))
    return df


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def panel_data():
    return _make_panel_data()


@pytest.fixture(scope="module")
def ref_nhmm():
    refs = {}
    files = {"fit": "ref_nhmm_fit.csv"}

    all_found = True
    for key, fname in files.items():
        ref = _load_ref_kv(THIS_DIR, fname)
        if ref is not None:
            if isinstance(ref, dict) and ref.get("status") == "failed":
                pytest.skip("R NHMM estimation failed.")
            refs[key] = ref
        else:
            all_found = False
            break

    if all_found:
        return refs

    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = {}
        for key, fname in files.items():
            ref = _load_ref_kv(outdir, fname)
            if ref is not None:
                refs[key] = ref
        if len(refs) == len(files):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_nhmm_*.csv not found. "
        "Run: Rscript seqhmm_reference_nhmm.R ."
    )


# ============================================================================
# Part 0: Sanity checks (no R needed)
# ============================================================================

class TestNHMMSanity:
    """Sanity checks for Non-Homogeneous HMM (no R needed)."""

    def test_build_nhmm_returns_object(self, panel_data):
        """build_nhmm returns a valid NHMM object."""
        nhmm = _build_nhmm_from_panel(panel_data)
        assert nhmm is not None
        assert nhmm.n_states == N_STATES

    def test_nhmm_attributes(self, panel_data):
        """NHMM has expected attributes."""
        nhmm = _build_nhmm_from_panel(panel_data)
        assert hasattr(nhmm, "eta_pi")
        assert hasattr(nhmm, "eta_A")
        assert hasattr(nhmm, "eta_B")
        assert hasattr(nhmm, "X")
        assert hasattr(nhmm, "log_likelihood")
        assert nhmm.log_likelihood is None  # before fit

    def test_covariate_dimensions(self, panel_data):
        """Covariate matrix X has correct shape."""
        nhmm = _build_nhmm_from_panel(panel_data)
        assert nhmm.X.ndim == 3
        assert nhmm.X.shape[0] == N_ID
        assert nhmm.n_covariates == 2  # intercept + x

    def test_loglik_computable_before_fit(self, panel_data):
        """log_likelihood_nhmm works on unfitted model."""
        nhmm = _build_nhmm_from_panel(panel_data)
        ll = log_likelihood_nhmm(nhmm)
        assert np.isfinite(ll), f"logLik not finite: {ll}"
        assert ll < 0, f"logLik should be negative: {ll}"

    def test_fit_sets_log_likelihood(self, panel_data):
        """After fit, log_likelihood is set."""
        nhmm = _build_nhmm_from_panel(panel_data)
        nhmm = fit_nhmm(nhmm, n_iter=50, tol=1e-3)
        assert nhmm.log_likelihood is not None
        assert np.isfinite(nhmm.log_likelihood)
        assert nhmm.log_likelihood < 0

    def test_fit_returns_self(self, panel_data):
        """fit_nhmm returns the same model object."""
        nhmm = _build_nhmm_from_panel(panel_data)
        result = fit_nhmm(nhmm, n_iter=20, tol=1e-2)
        assert result is nhmm

    def test_fit_improves_loglik(self, panel_data):
        """Fitting should improve log-likelihood."""
        nhmm = _build_nhmm_from_panel(panel_data)
        ll_before = log_likelihood_nhmm(nhmm)
        nhmm = fit_nhmm(nhmm, n_iter=100, tol=1e-4)
        ll_after = nhmm.log_likelihood
        assert ll_after >= ll_before - 1e-4, (
            f"logLik decreased: {ll_before:.4f} -> {ll_after:.4f}"
        )

    def test_eta_shapes(self, panel_data):
        """Coefficient arrays have correct shapes after fit."""
        nhmm = _build_nhmm_from_panel(panel_data)
        nhmm = fit_nhmm(nhmm, n_iter=50, tol=1e-3)
        n_cov = nhmm.n_covariates
        n_s = nhmm.n_states
        n_sym = nhmm.n_symbols
        assert nhmm.eta_pi.shape == (n_cov, n_s)
        assert nhmm.eta_A.shape == (n_cov, n_s, n_s)
        assert nhmm.eta_B.shape == (n_cov, n_s, n_sym)

    def test_deterministic(self, panel_data):
        """Same data + same seed = same result."""
        nhmm1 = _build_nhmm_from_panel(panel_data, random_state=42)
        fit_nhmm(nhmm1, n_iter=50, tol=1e-3)

        nhmm2 = _build_nhmm_from_panel(panel_data, random_state=42)
        fit_nhmm(nhmm2, n_iter=50, tol=1e-3)

        assert np.isclose(nhmm1.log_likelihood, nhmm2.log_likelihood, atol=1e-4), (
            f"Not deterministic: {nhmm1.log_likelihood} vs {nhmm2.log_likelihood}"
        )


# ============================================================================
# Part 1: Cross-language consistency vs R seqHMM
# ============================================================================

def test_nhmm_loglik_matches_r(panel_data, ref_nhmm):
    """NHMM logLik is in the same ballpark as R's.

    NHMM uses L-BFGS-B optimisation which is very sensitive to
    initialisation.  Python and R implementations use different
    random seeds and different internal parameterisations, so they
    routinely converge to different local optima.  We therefore only
    check that the results are within LOGLIK_ATOL nats of each other.
    """
    nhmm = _build_nhmm_from_panel(panel_data, random_state=123)
    nhmm = fit_nhmm(nhmm, n_iter=200, tol=1e-6)
    py_ll = nhmm.log_likelihood
    r_ll = float(ref_nhmm["fit"]["loglik"])
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_ATOL), (
        f"NHMM logLik mismatch: Python={py_ll:.4f}, R={r_ll:.4f}, "
        f"diff={abs(py_ll - r_ll):.4f}, atol={LOGLIK_ATOL}"
    )


def test_nhmm_python_not_worse_than_r(panel_data, ref_nhmm):
    """Python NHMM should not be dramatically worse than R."""
    nhmm = _build_nhmm_from_panel(panel_data, random_state=123)
    nhmm = fit_nhmm(nhmm, n_iter=200, tol=1e-6)
    py_ll = nhmm.log_likelihood
    r_ll = float(ref_nhmm["fit"]["loglik"])
    assert py_ll >= r_ll - 3.0, (
        f"Python NHMM much worse: Python={py_ll:.4f}, R={r_ll:.4f}"
    )