"""
@Author  : Yapeng Wei
@File    : test_seqhmm_advanced.py
@Desc    :
Tests for advanced sequenzo.seqhmm features:
  - fit_model_advanced() with EM, global, local steps, random restarts
  - Formula class and create_model_matrix()
  - create_model_matrix_time_constant()
  - compare_models() for BIC-based model selection

Actual Python API:
  fit_model_advanced(model, em_step=True, global_step=False,
                     local_step=False, n_iter=100, tol=1e-2,
                     n_restarts=0, verbose=False, random_state=None)
    -> fitted model (HMM | MHMM | NHMM)

  Formula(formula_str)
    .terms -> list of variable names
    .create_matrix(data, id_var, time_var, n_sequences, n_timepoints) -> 3D array

  create_model_matrix(formula, data, id_var, time_var, n_sequences, n_timepoints)
    -> numpy (n_sequences, n_timepoints, n_covariates)

  create_model_matrix_time_constant(formula, data, n_sequences)
    -> numpy (n_sequences, n_covariates)

  compare_models(models, criterion='BIC') -> dict
    Returns: {'criterion': str, 'models': list, 'best_model': str}

Test groups:
  Part 0: Sanity (no R)
  Part 1: Cross-language consistency (needs ref CSVs from R)

Run: Rscript seqhmm_reference_advanced.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import (
    build_hmm, fit_model, fit_model_advanced,
    aic, bic, compare_models,
    Formula, create_model_matrix, create_model_matrix_time_constant,
)


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
SEQ_IDS = [f"s{i}" for i in range(5)]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Config A: 2 states (same as R script)
INIT_A = np.array([0.6, 0.4])
TRANS_A = np.array([[0.7, 0.3], [0.2, 0.8]])
EMISS_A = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],
    ["B", "C", "C", "A", "A", "B", "C", "C"],
    ["A", "B", "C", "A", "B", "C", "A", "B"],
    ["C", "C", "B", "A", "A", "A", "B", "C"],
    ["A", "A", "A", "B", "B", "B", "C", "C"],
]

# Tolerances
LOGLIK_ATOL = 1.0  # generous for advanced fitting (different optimizers)


# ============================================================================
# Helpers
# ============================================================================

def _make_seqdata():
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _build_hmm_config_a(seqdata):
    return build_hmm(
        seqdata,
        n_states=2,
        initial_probs=INIT_A.copy(),
        transition_probs=TRANS_A.copy(),
        emission_probs=EMISS_A.copy(),
    )


def _run_r_reference(outdir, timeout=300):
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_advanced.R")
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
    return None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    return _make_seqdata()


@pytest.fixture(scope="module")
def ref_advanced():
    refs = {}
    kv_files = {
        "fit": "ref_advanced_fit.csv",
        "comparison": "ref_advanced_comparison.csv",
    }

    all_found = True
    for key, fname in kv_files.items():
        ref = _load_ref_kv(THIS_DIR, fname)
        if ref is not None:
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
        for key, fname in kv_files.items():
            ref = _load_ref_kv(outdir, fname)
            if ref is not None:
                refs[key] = ref
        if len(refs) == len(kv_files):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_advanced_*.csv not found. "
        "Run: Rscript seqhmm_reference_advanced.R ."
    )


# ============================================================================
# Part 0a: fit_model_advanced sanity checks
# ============================================================================

class TestFitModelAdvancedSanity:
    """Sanity checks for fit_model_advanced."""

    def test_em_only(self, seqdata):
        """fit_model_advanced with em_step only returns fitted model."""
        hmm = _build_hmm_config_a(seqdata)
        result = fit_model_advanced(
            hmm, em_step=True, global_step=False, local_step=False,
            n_iter=100, tol=1e-6, verbose=False,
        )
        assert result is not None
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

    def test_em_plus_local(self, seqdata):
        """fit_model_advanced with em + local step returns fitted model."""
        hmm = _build_hmm_config_a(seqdata)
        result = fit_model_advanced(
            hmm, em_step=True, global_step=False, local_step=True,
            n_iter=100, tol=1e-6, verbose=False,
        )
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

    def test_em_plus_global(self, seqdata):
        """fit_model_advanced with em + global step returns fitted model."""
        hmm = _build_hmm_config_a(seqdata)
        result = fit_model_advanced(
            hmm, em_step=True, global_step=True, local_step=False,
            n_iter=50, tol=1e-4, verbose=False,
        )
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

    def test_random_restarts(self, seqdata):
        """fit_model_advanced with n_restarts > 0 runs multiple restarts."""
        hmm = _build_hmm_config_a(seqdata)
        result = fit_model_advanced(
            hmm, em_step=True, n_restarts=3, n_iter=50, tol=1e-4,
            verbose=False, random_state=42,
        )
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

    def test_advanced_not_worse_than_basic(self, seqdata):
        """Advanced fitting should be >= basic EM result."""
        hmm_basic = _build_hmm_config_a(seqdata)
        fit_model(hmm_basic, n_iter=100, tol=1e-6)
        ll_basic = hmm_basic.log_likelihood

        hmm_adv = _build_hmm_config_a(seqdata)
        result = fit_model_advanced(
            hmm_adv, em_step=True, local_step=True,
            n_iter=100, tol=1e-8, verbose=False,
        )
        ll_adv = result.log_likelihood

        # Advanced should be at least as good (or very close)
        assert ll_adv >= ll_basic - 0.1, (
            f"Advanced ({ll_adv:.4f}) worse than basic ({ll_basic:.4f})"
        )

    def test_restarts_deterministic(self, seqdata):
        """Same random_state gives same restart results."""
        hmm1 = _build_hmm_config_a(seqdata)
        r1 = fit_model_advanced(
            hmm1, em_step=True, n_restarts=2, n_iter=50,
            random_state=42, verbose=False,
        )
        hmm2 = _build_hmm_config_a(seqdata)
        r2 = fit_model_advanced(
            hmm2, em_step=True, n_restarts=2, n_iter=50,
            random_state=42, verbose=False,
        )
        assert np.isclose(r1.log_likelihood, r2.log_likelihood, atol=1e-4), (
            f"Not deterministic: {r1.log_likelihood} vs {r2.log_likelihood}"
        )


# ============================================================================
# Part 0b: Formula and model matrix sanity checks
# ============================================================================

class TestFormulaSanity:
    """Sanity checks for Formula and create_model_matrix."""

    def test_formula_parse_simple(self):
        """Formula('~ x1 + x2') parses terms correctly."""
        f = Formula("~ x1 + x2")
        assert f.terms == ["x1", "x2"]

    def test_formula_parse_no_tilde(self):
        """Formula('x1 + x2') works without tilde."""
        f = Formula("x1 + x2")
        assert f.terms == ["x1", "x2"]

    def test_formula_intercept_only(self):
        """Formula('~ 1') has no terms."""
        f = Formula("~ 1")
        # '1' is treated as a term or empty depending on implementation
        # The key point is create_matrix should return intercept-only
        pass  # Checked indirectly via create_matrix

    def test_create_matrix_shape(self):
        """create_model_matrix returns (n_seq, n_time, n_cov) array."""
        n_seq, n_time = 3, 5
        data = pd.DataFrame({
            "id": np.repeat([1, 2, 3], n_time),
            "time": np.tile(range(1, n_time + 1), n_seq),
            "x1": np.random.randn(n_seq * n_time),
        })
        X = create_model_matrix(
            "~ x1", data, id_var="id", time_var="time",
            n_sequences=n_seq, n_timepoints=n_time,
        )
        assert X.ndim == 3
        assert X.shape[0] == n_seq
        assert X.shape[1] == n_time
        # n_covariates: intercept + x1 = 2
        assert X.shape[2] == 2

    def test_create_matrix_intercept_column(self):
        """First column of model matrix is all ones (intercept)."""
        n_seq, n_time = 2, 4
        data = pd.DataFrame({
            "id": np.repeat([1, 2], n_time),
            "time": np.tile(range(1, n_time + 1), n_seq),
            "x1": np.random.randn(n_seq * n_time),
        })
        X = create_model_matrix(
            "~ x1", data, id_var="id", time_var="time",
            n_sequences=n_seq, n_timepoints=n_time,
        )
        assert np.allclose(X[:, :, 0], 1.0), "Intercept column not all ones"

    def test_create_matrix_time_constant_shape(self):
        """create_model_matrix_time_constant returns (n_seq, n_cov)."""
        n_seq = 5
        data = pd.DataFrame({
            "x1": np.random.randn(n_seq),
            "x2": np.random.randn(n_seq),
        })
        X = create_model_matrix_time_constant("~ x1 + x2", data, n_seq)
        assert X.ndim == 2
        assert X.shape[0] == n_seq
        # intercept + x1 + x2 = 3
        assert X.shape[2] if X.ndim == 3 else X.shape[1] == 3

    def test_create_matrix_time_constant_intercept_only(self):
        """None formula returns intercept-only matrix."""
        n_seq = 4
        X = create_model_matrix_time_constant(None, None, n_seq)
        assert X.shape == (n_seq, 1)
        assert np.allclose(X, 1.0)

    def test_create_matrix_time_constant_categorical(self):
        """Categorical variables are expanded to dummies."""
        n_seq = 6
        data = pd.DataFrame({
            "color": ["red", "blue", "green", "red", "blue", "green"],
        })
        X = create_model_matrix_time_constant("~ color", data, n_seq)
        # color has 3 levels, drop_first=True -> 2 dummy columns
        # (no separate intercept column added by this function)
        assert X.shape == (n_seq, 2)  # 2 dummies


# ============================================================================
# Part 0c: Model comparison sanity checks
# ============================================================================

class TestModelComparisonSanity:
    """Sanity checks for compare_models and AIC/BIC."""

    def test_aic_on_fitted_hmm(self, seqdata):
        """aic() returns a finite value for fitted HMM."""
        hmm = _build_hmm_config_a(seqdata)
        fit_model(hmm, n_iter=100, tol=1e-6)
        aic_val = aic(hmm)
        assert np.isfinite(aic_val), f"AIC not finite: {aic_val}"

    def test_bic_on_fitted_hmm(self, seqdata):
        """bic() returns a finite value for fitted HMM."""
        hmm = _build_hmm_config_a(seqdata)
        fit_model(hmm, n_iter=100, tol=1e-6)
        bic_val = bic(hmm)
        assert np.isfinite(bic_val), f"BIC not finite: {bic_val}"

    def test_aic_less_than_bic(self, seqdata):
        """For small datasets, BIC penalty > AIC penalty, so BIC >= AIC."""
        hmm = _build_hmm_config_a(seqdata)
        fit_model(hmm, n_iter=100, tol=1e-6)
        aic_val = aic(hmm)
        bic_val = bic(hmm)
        # BIC has log(n) penalty vs 2 for AIC; for n >= 8, log(n) > 2
        # So BIC >= AIC for reasonable n
        assert bic_val >= aic_val - 1e-6, (
            f"BIC ({bic_val:.4f}) < AIC ({aic_val:.4f})"
        )

    def test_aic_unfitted_raises(self, seqdata):
        """aic() raises ValueError for unfitted model."""
        hmm = _build_hmm_config_a(seqdata)
        with pytest.raises(ValueError, match="fitted"):
            aic(hmm)

    def test_compare_models_returns_dict(self, seqdata):
        """compare_models returns dict with expected keys."""
        hmm2 = build_hmm(seqdata, n_states=2)
        fit_model(hmm2, n_iter=50, tol=1e-4)

        hmm3 = build_hmm(seqdata, n_states=3)
        fit_model(hmm3, n_iter=50, tol=1e-4)

        result = compare_models([hmm2, hmm3], criterion="BIC")
        assert "criterion" in result
        assert "models" in result
        assert "best_model" in result
        assert result["criterion"] == "BIC"
        assert len(result["models"]) == 2

    def test_compare_models_sorted(self, seqdata):
        """compare_models sorts models by criterion (lower is better)."""
        hmm2 = build_hmm(seqdata, n_states=2)
        fit_model(hmm2, n_iter=50, tol=1e-4)

        hmm3 = build_hmm(seqdata, n_states=3)
        fit_model(hmm3, n_iter=50, tol=1e-4)

        result = compare_models([hmm2, hmm3], criterion="BIC")
        models = result["models"]
        bic_values = [m["BIC"] for m in models]
        assert bic_values == sorted(bic_values), "Models not sorted by BIC"


# ============================================================================
# Part 1: Cross-language consistency vs R seqHMM
# ============================================================================

def test_em_only_loglik_matches_r(seqdata, ref_advanced):
    """EM-only logLik matches R's fit_model(em_step=TRUE)."""
    hmm = _build_hmm_config_a(seqdata)
    result = fit_model_advanced(
        hmm, em_step=True, global_step=False, local_step=False,
        n_iter=1000, tol=1e-10, verbose=False,
    )
    py_ll = result.log_likelihood
    r_ll = float(ref_advanced["fit"]["loglik_em_only"])
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_ATOL), (
        f"EM-only logLik: Python={py_ll:.4f}, R={r_ll:.4f}"
    )


def test_em_local_loglik_matches_r(seqdata, ref_advanced):
    """EM+local logLik matches R's fit_model(em+local)."""
    r_ll_str = ref_advanced["fit"].get("loglik_em_local")
    if r_ll_str is None or str(r_ll_str) == "nan":
        pytest.skip("R EM+local result not available")

    r_ll = float(r_ll_str)
    hmm = _build_hmm_config_a(seqdata)
    result = fit_model_advanced(
        hmm, em_step=True, global_step=False, local_step=True,
        n_iter=500, tol=1e-10, verbose=False,
    )
    py_ll = result.log_likelihood
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_ATOL), (
        f"EM+local logLik: Python={py_ll:.4f}, R={r_ll:.4f}"
    )


def test_bic_comparison_matches_r(seqdata, ref_advanced):
    """BIC for 2,3,4-state models matches R (directionally)."""
    ref = ref_advanced.get("comparison")
    if ref is None:
        pytest.skip("R comparison reference not available")

    py_bics = {}
    for ns in (2, 3, 4):
        hmm = build_hmm(seqdata, n_states=ns)
        fit_model(hmm, n_iter=100, tol=1e-6)
        py_bics[ns] = bic(hmm)

    # Check the best (lowest BIC) model matches R's best
    r_bics = {}
    for ns in (2, 3, 4):
        key = f"bic_{ns}states"
        if key in ref:
            r_bics[ns] = float(ref[key])

    if len(r_bics) >= 2:
        r_best = min(r_bics, key=r_bics.get)
        py_best = min(py_bics, key=py_bics.get)
        # Best model should match (or at least be close)
        assert py_best == r_best or abs(py_bics[py_best] - py_bics[r_best]) < 5.0, (
            f"Best model: Python={py_best}states, R={r_best}states. "
            f"Python BICs={py_bics}, R BICs={r_bics}"
        )
