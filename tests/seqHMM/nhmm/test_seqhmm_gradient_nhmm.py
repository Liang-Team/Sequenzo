"""
@Author  : Yapeng Wei
@File    : test_seqhmm_gradient_nhmm.py
@Desc    :
Tests for compute_gradient_nhmm() in sequenzo.seqhmm.

Actual Python API:
  compute_gradient_nhmm(model: NHMM) -> np.ndarray
    Input: A fitted (or at least parameterised) NHMM model with:
      .eta_pi, .eta_A, .eta_B, .X, .observations,
      .n_states, .n_symbols, ._compute_probs()
    Output: Flattened gradient vector [grad_eta_pi, grad_eta_A, grad_eta_B]

The gradient is used internally by fit_nhmm (L-BFGS-B) to optimize
NHMM log-likelihood.  We validate it via finite-difference checks:
  numerical_grad ≈ analytical_grad.

Test groups:
  Part 0: Sanity checks on gradient properties
  Part 1: Finite-difference verification
"""
import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_nhmm, fit_nhmm
from sequenzo.seqhmm.gradients_nhmm import compute_gradient_nhmm
from sequenzo.seqhmm.forward_backward_nhmm import log_likelihood_nhmm


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
N_STATES = 2
N_ID = 10
N_TIME = 8
FD_EPS = 1e-5          # finite-difference step
FD_RTOL = 0.10         # 10 % relative tolerance (analytical vs numerical)
FD_ATOL = 1e-3         # absolute tolerance for near-zero components


# ============================================================================
# Helpers
# ============================================================================

def _make_panel_and_nhmm(random_state=42):
    """Build a small NHMM for testing (NOT fitted, just parameterised)."""
    rng = np.random.RandomState(random_state)
    ids = np.repeat(np.arange(1, N_ID + 1), N_TIME)
    times = np.tile(np.arange(1, N_TIME + 1), N_ID)
    x = np.round(rng.randn(N_ID * N_TIME), 3)
    response = rng.choice(STATES, size=N_ID * N_TIME)

    panel = pd.DataFrame({
        "id": ids, "time": times, "response": response, "x": x,
    })

    # Pivot to wide format for SequenceData
    wide = panel.pivot(index="id", columns="time", values="response")
    wide = wide.reset_index()
    time_cols = [c for c in wide.columns if c != "id"]
    seqdata = SequenceData(wide, time=time_cols, states=STATES, id_col="id")

    # Build covariate matrix X: (n_seq, n_time, n_cov) with intercept + x
    x_wide = panel.pivot(index="id", columns="time", values="x").values
    intercept = np.ones((N_ID, N_TIME, 1))
    x_3d = np.stack([np.ones((N_ID, N_TIME)), x_wide], axis=-1)

    nhmm = build_nhmm(
        observations=seqdata,
        n_states=N_STATES,
        X=x_3d,
        random_state=random_state,
    )
    return nhmm


def _perturb_and_loglik(nhmm, param_name, flat_idx, eps):
    """Perturb one element of eta by ±eps and return the two log-likelihoods."""
    eta = getattr(nhmm, param_name).copy()
    orig_shape = eta.shape
    flat = eta.flatten()

    # +eps
    flat_plus = flat.copy()
    flat_plus[flat_idx] += eps
    setattr(nhmm, param_name, flat_plus.reshape(orig_shape))
    ll_plus = log_likelihood_nhmm(nhmm)

    # -eps
    flat_minus = flat.copy()
    flat_minus[flat_idx] -= eps
    setattr(nhmm, param_name, flat_minus.reshape(orig_shape))
    ll_minus = log_likelihood_nhmm(nhmm)

    # Restore
    setattr(nhmm, param_name, eta)
    return ll_plus, ll_minus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def nhmm():
    return _make_panel_and_nhmm(random_state=42)


@pytest.fixture(scope="module")
def fitted_nhmm():
    model = _make_panel_and_nhmm(random_state=42)
    return fit_nhmm(model, n_iter=30, tol=1e-4, verbose=False)


# ============================================================================
# Part 0: Sanity checks
# ============================================================================

class TestGradientSanity:
    """Basic sanity checks for compute_gradient_nhmm."""

    def test_returns_1d_array(self, nhmm):
        """Gradient is a 1-D numpy array."""
        grad = compute_gradient_nhmm(nhmm)
        assert isinstance(grad, np.ndarray)
        assert grad.ndim == 1

    def test_correct_length(self, nhmm):
        """Gradient vector length == total number of parameters."""
        grad = compute_gradient_nhmm(nhmm)
        n_cov = nhmm.X.shape[2]
        n_sym = nhmm.n_symbols
        n_st = nhmm.n_states
        expected = n_cov * n_st + n_cov * n_st * n_st + n_cov * n_st * n_sym
        assert len(grad) == expected, (
            f"Gradient length {len(grad)} != expected {expected}"
        )

    def test_no_nan(self, nhmm):
        """Gradient has no NaN values."""
        grad = compute_gradient_nhmm(nhmm)
        assert not np.any(np.isnan(grad)), "Gradient contains NaN"

    def test_no_inf(self, nhmm):
        """Gradient has no Inf values."""
        grad = compute_gradient_nhmm(nhmm)
        assert not np.any(np.isinf(grad)), "Gradient contains Inf"

    def test_gradient_not_all_zero(self, nhmm):
        """Unfitted model should have a non-zero gradient."""
        grad = compute_gradient_nhmm(nhmm)
        assert np.any(np.abs(grad) > 1e-10), "Gradient is all zeros"

    def test_gradient_small_at_optimum(self, fitted_nhmm):
        """Gradient norm should be relatively small at a fitted optimum."""
        grad = compute_gradient_nhmm(fitted_nhmm)
        grad_norm = np.linalg.norm(grad)
        # After fitting, gradient should be smaller than at random init
        # (not necessarily zero due to approximate convergence)
        assert grad_norm < 100, (
            f"Gradient norm {grad_norm:.2f} unexpectedly large at fitted model"
        )

    def test_deterministic(self, nhmm):
        """Same model gives same gradient."""
        g1 = compute_gradient_nhmm(nhmm)
        g2 = compute_gradient_nhmm(nhmm)
        np.testing.assert_array_equal(g1, g2)


# ============================================================================
# Part 1: Finite-difference verification
# ============================================================================

class TestGradientFiniteDifference:
    """Verify analytical gradient against numerical finite differences.

    For each parameter group (eta_pi, eta_A, eta_B), we perturb a subset
    of elements by ±eps and check that:
        (LL(+eps) - LL(-eps)) / (2*eps)  ≈  analytical gradient
    """

    @staticmethod
    def _check_fd(nhmm_model, param_name, n_checks=5):
        """Run finite-difference checks on a subset of elements."""
        grad_full = compute_gradient_nhmm(nhmm_model)
        eta = getattr(nhmm_model, param_name)
        n_param = eta.size

        # Determine offset into the flattened gradient vector
        n_cov = nhmm_model.X.shape[2]
        n_st = nhmm_model.n_states
        n_sym = nhmm_model.n_symbols
        if param_name == "eta_pi":
            offset = 0
        elif param_name == "eta_A":
            offset = n_cov * n_st
        else:  # eta_B
            offset = n_cov * n_st + n_cov * n_st * n_st

        rng = np.random.RandomState(0)
        indices = rng.choice(n_param, size=min(n_checks, n_param), replace=False)

        mismatches = []
        for idx in indices:
            ll_plus, ll_minus = _perturb_and_loglik(
                nhmm_model, param_name, idx, FD_EPS
            )
            num_grad = (ll_plus - ll_minus) / (2 * FD_EPS)
            ana_grad = grad_full[offset + idx]

            if abs(num_grad) < FD_ATOL and abs(ana_grad) < FD_ATOL:
                continue  # both near zero, skip
            if not np.isclose(num_grad, ana_grad, rtol=FD_RTOL, atol=FD_ATOL):
                mismatches.append(
                    f"  {param_name}[{idx}]: numerical={num_grad:.6f}, "
                    f"analytical={ana_grad:.6f}, "
                    f"rel_err={abs(num_grad - ana_grad) / (abs(num_grad) + 1e-10):.4f}"
                )
        return mismatches

    def test_fd_eta_pi(self, nhmm):
        """Finite-difference check for eta_pi (initial probs)."""
        mismatches = self._check_fd(nhmm, "eta_pi", n_checks=4)
        assert len(mismatches) == 0, (
            f"eta_pi gradient mismatch:\n" + "\n".join(mismatches)
        )

    def test_fd_eta_A(self, nhmm):
        """Finite-difference check for eta_A (transition probs)."""
        mismatches = self._check_fd(nhmm, "eta_A", n_checks=6)
        assert len(mismatches) == 0, (
            f"eta_A gradient mismatch:\n" + "\n".join(mismatches)
        )

    def test_fd_eta_B(self, nhmm):
        """Finite-difference check for eta_B (emission probs)."""
        mismatches = self._check_fd(nhmm, "eta_B", n_checks=6)
        assert len(mismatches) == 0, (
            f"eta_B gradient mismatch:\n" + "\n".join(mismatches)
        )
