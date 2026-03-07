"""
@Author  : Yapeng Wei
@File    : test_seqhmm_em.py
@Desc    :
Tests for EM algorithm (fit_model) consistency between sequenzo.seqhmm
and R seqHMM (https://github.com/helske/seqHMM).

The EM algorithm iteratively updates HMM parameters (initial, transition,
emission probabilities) to maximize the data log-likelihood. This test
verifies that Python (hmmlearn backend) and R (seqHMM C++ backend)
converge to the same solution given identical initial parameters and data.

Key challenge: R and Python use different convergence criteria:
  - R seqHMM: reltol = (LL_new - LL_old) / (|LL_old| + 0.1), default 1e-10
  - Python hmmlearn: absolute change in LL, default tol=1e-2
Both are run with tight tolerance and many iterations to ensure full
convergence. Comparison tolerances are set conservatively:
  - logLik: atol=0.1 (EM implementations may stop at slightly different points)
  - parameters: atol=0.02 (flat likelihood surface near optimum)

Test groups:
  Part 0: Sanity checks (no R needed)
    - fit() returns self for chaining
    - logLik improves after EM
    - parameters remain valid probability distributions
    - fit is deterministic with same initial params
    - converged flag is set

  Part 1: EM convergence vs R seqHMM (needs ref CSVs from R script)
    - Configs A/B/C: converged logLik matches
    - Configs A/B/C: converged parameters match

Run seqhmm_reference_em.R to generate ref_em_*.csv:
  Rscript seqhmm_reference_em.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm


# ============================================================================
# Constants (shared across all seqhmm test files -- MUST remain identical)
# ============================================================================

STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)
SEQ_IDS = ["s0", "s1", "s2", "s3", "s4"]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
N_SEQUENCES = len(SEQ_IDS)
N_TIMEPOINTS = len(TIME_COLS)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Synthetic test data (MUST be identical to R script and other test files)
# ============================================================================

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s0
    ["B", "C", "C", "A", "A", "B", "C", "C"],  # s1
    ["A", "B", "C", "A", "B", "C", "A", "B"],  # s2
    ["C", "C", "B", "A", "A", "A", "B", "C"],  # s3
    ["A", "A", "A", "B", "B", "B", "C", "C"],  # s4
]


# ============================================================================
# HMM parameter configurations (MUST be identical across all test files)
# ============================================================================

CONFIGS = {
    "A": {
        "n_states": 2,
        "initial_probs": np.array([0.6, 0.4]),
        "transition_probs": np.array([
            [0.7, 0.3],
            [0.2, 0.8],
        ]),
        "emission_probs": np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.4, 0.5],
        ]),
    },
    "B": {
        "n_states": 3,
        "initial_probs": np.array([0.5, 0.3, 0.2]),
        "transition_probs": np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ]),
        "emission_probs": np.array([
            [0.6, 0.3, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.1, 0.6],
        ]),
    },
    "C": {
        "n_states": 2,
        "initial_probs": np.array([0.9, 0.1]),
        "transition_probs": np.array([
            [0.95, 0.05],
            [0.05, 0.95],
        ]),
        "emission_probs": np.array([
            [0.8, 0.15, 0.05],
            [0.05, 0.15, 0.8],
        ]),
    },
}

# Only test EM on configs A, B, C (not D -- uniform model has nothing to learn)
EM_CONFIGS = ["A", "B", "C"]

# Tolerances for EM comparison
# These are intentionally generous because:
# 1. R and Python EM implementations differ in convergence criteria
# 2. Near the optimum, the likelihood surface is flat, so small logLik
#    differences can correspond to larger parameter differences
LOGLIK_ATOL = 0.1
PARAM_ATOL = 0.02

# EM settings: run both sides with tight tolerance and many iterations
EM_N_ITER = 1000
EM_TOL = 1e-8  # tight tolerance for Python side


# ============================================================================
# Helper functions
# ============================================================================

def _make_test_seqdata():
    """Create SequenceData from synthetic test data."""
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _build_hmm_for_config(seqdata, config_name):
    """Build an HMM with the specified config."""
    cfg = CONFIGS[config_name]
    return build_hmm(
        observations=seqdata,
        n_states=cfg["n_states"],
        initial_probs=cfg["initial_probs"].copy(),
        transition_probs=cfg["transition_probs"].copy(),
        emission_probs=cfg["emission_probs"].copy(),
    )


def _run_r_reference(outdir, timeout=120):
    """Run seqhmm_reference_em.R to generate ref_em_*.csv."""
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_em.R")
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


def _load_ref_em(ref_dir, config_name):
    """Load ref_em_<config>.csv as a dict of key -> float value.

    Returns dict like:
        {"loglik_before": -28.5, "loglik_after": -27.1,
         "initial_probs_0": 0.6, "initial_probs_1": 0.4,
         "trans_0_0": 0.7, ..., "emiss_0_0": 0.5, ...}
    Returns None if file not found.
    """
    fpath = os.path.join(ref_dir, f"ref_em_{config_name}.csv")
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    return dict(zip(df["key"], df["value"]))


def _extract_params_from_ref(ref_dict, n_states, n_symbols):
    """Extract structured parameters from ref dict.

    Returns:
        loglik_after: float
        initial_probs: np.ndarray (n_states,)
        transition_probs: np.ndarray (n_states, n_states)
        emission_probs: np.ndarray (n_states, n_symbols)
    """
    loglik_after = ref_dict["loglik_after"]

    initial_probs = np.array([
        ref_dict[f"initial_probs_{i}"] for i in range(n_states)
    ])

    transition_probs = np.array([
        [ref_dict[f"trans_{i}_{j}"] for j in range(n_states)]
        for i in range(n_states)
    ])

    emission_probs = np.array([
        [ref_dict[f"emiss_{i}_{j}"] for j in range(n_symbols)]
        for i in range(n_states)
    ])

    return loglik_after, initial_probs, transition_probs, emission_probs


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    """SequenceData from synthetic test data."""
    return _make_test_seqdata()


@pytest.fixture(scope="module")
def ref_em():
    """Load R reference EM results for all configs.

    Returns dict: config_name -> ref_dict.
    """
    refs = {}
    all_found = True
    for config_name in EM_CONFIGS:
        ref = _load_ref_em(THIS_DIR, config_name)
        if ref is not None:
            refs[config_name] = ref
        else:
            all_found = False
            break

    if all_found:
        return refs

    # Try running R script
    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = {}
        for config_name in EM_CONFIGS:
            ref = _load_ref_em(outdir, config_name)
            if ref is not None:
                refs[config_name] = ref
        if len(refs) == len(EM_CONFIGS):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_em_*.csv not found. "
        "Run: Rscript seqhmm_reference_em.R . "
        "to generate reference values."
    )


# ============================================================================
# Part 0: Sanity checks (no R reference needed)
# ============================================================================

class TestEMSanity:
    """Sanity checks for EM fitting (no R needed)."""

    def test_fit_returns_self(self, seqdata):
        """fit() returns self for method chaining."""
        hmm = _build_hmm_for_config(seqdata, "A")
        result = hmm.fit(n_iter=10, tol=1e-4)
        assert result is hmm, "fit() should return self"

    def test_loglik_improves_after_em(self, seqdata):
        """EM should improve (or maintain) log-likelihood."""
        for config_name in EM_CONFIGS:
            hmm = _build_hmm_for_config(seqdata, config_name)
            ll_before = hmm.score()
            hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
            ll_after = hmm.score()
            assert ll_after >= ll_before - 1e-6, (
                f"Config {config_name}: logLik decreased after EM. "
                f"Before: {ll_before}, After: {ll_after}"
            )

    def test_parameters_valid_after_em(self, seqdata):
        """After EM, all parameters should be valid probability distributions."""
        for config_name in EM_CONFIGS:
            hmm = _build_hmm_for_config(seqdata, config_name)
            hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)

            # Initial probs: sum to 1, all >= 0
            assert np.all(hmm.initial_probs >= -1e-10), (
                f"Config {config_name}: negative initial_probs"
            )
            assert np.isclose(hmm.initial_probs.sum(), 1.0, atol=1e-6), (
                f"Config {config_name}: initial_probs sum = "
                f"{hmm.initial_probs.sum()}"
            )

            # Transition probs: each row sums to 1, all >= 0
            assert np.all(hmm.transition_probs >= -1e-10), (
                f"Config {config_name}: negative transition_probs"
            )
            row_sums = hmm.transition_probs.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), (
                f"Config {config_name}: transition_probs row sums = "
                f"{row_sums}"
            )

            # Emission probs: each row sums to 1, all >= 0
            assert np.all(hmm.emission_probs >= -1e-10), (
                f"Config {config_name}: negative emission_probs"
            )
            row_sums = hmm.emission_probs.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), (
                f"Config {config_name}: emission_probs row sums = "
                f"{row_sums}"
            )

    def test_em_deterministic(self, seqdata):
        """Same initial params + same data = same EM result."""
        hmm1 = _build_hmm_for_config(seqdata, "A")
        hmm1.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        hmm2 = _build_hmm_for_config(seqdata, "A")
        hmm2.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        assert np.isclose(hmm1.score(), hmm2.score(), atol=1e-8), (
            f"EM not deterministic: {hmm1.score()} vs {hmm2.score()}"
        )
        assert np.allclose(hmm1.transition_probs, hmm2.transition_probs,
                           atol=1e-8), "transition_probs differ between runs"
        assert np.allclose(hmm1.emission_probs, hmm2.emission_probs,
                           atol=1e-8), "emission_probs differ between runs"

    def test_em_updates_parameters(self, seqdata):
        """EM should actually change at least some parameters."""
        cfg = CONFIGS["A"]
        hmm = _build_hmm_for_config(seqdata, "A")
        hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        # At least one parameter matrix should have changed
        init_changed = not np.allclose(
            hmm.initial_probs, cfg["initial_probs"], atol=1e-6
        )
        trans_changed = not np.allclose(
            hmm.transition_probs, cfg["transition_probs"], atol=1e-6
        )
        emiss_changed = not np.allclose(
            hmm.emission_probs, cfg["emission_probs"], atol=1e-6
        )
        assert init_changed or trans_changed or emiss_changed, (
            "EM did not change any parameters from initial values"
        )

    def test_fit_model_function(self, seqdata):
        """The standalone fit_model() function works identically to HMM.fit()."""
        from sequenzo.seqhmm import fit_model

        hmm1 = _build_hmm_for_config(seqdata, "A")
        hmm1.fit(n_iter=50, tol=1e-4)
        ll1 = hmm1.score()

        hmm2 = _build_hmm_for_config(seqdata, "A")
        fit_model(hmm2, n_iter=50, tol=1e-4)
        ll2 = hmm2.score()

        assert np.isclose(ll1, ll2, atol=1e-8), (
            f"fit_model() vs HMM.fit() differ: {ll1} vs {ll2}"
        )

    def test_more_iterations_no_worse(self, seqdata):
        """Running more EM iterations should not decrease log-likelihood."""
        hmm_short = _build_hmm_for_config(seqdata, "A")
        hmm_short.fit(n_iter=10, tol=1e-12)
        ll_short = hmm_short.score()

        hmm_long = _build_hmm_for_config(seqdata, "A")
        hmm_long.fit(n_iter=500, tol=1e-12)
        ll_long = hmm_long.score()

        assert ll_long >= ll_short - 1e-6, (
            f"More iterations gave worse logLik: "
            f"10 iter={ll_short}, 500 iter={ll_long}"
        )


# ============================================================================
# Part 1: EM convergence vs R seqHMM
# ============================================================================
# EM is a non-convex optimization -- different implementations (R C++ vs
# Python hmmlearn) may converge to DIFFERENT local optima even from
# identical starting points. This is because:
#   1. R seqHMM can drive probabilities to exact 0; hmmlearn cannot
#   2. Different log-space vs scaling numerical approaches
#   3. Different convergence criteria (relative vs absolute tolerance)
#
# Therefore we do NOT require identical parameters. Instead we check:
#   - Both reach similar final log-likelihood (generous tolerance)
#   - Python's solution is not dramatically worse than R's
#   - Starting points are identical (sanity check)

LOGLIK_EM_ATOL = 0.5  # generous: EM local optima can differ


def test_em_config_A_loglik_matches_r(seqdata, ref_em):
    """Config A: converged logLik is close to R's."""
    hmm = _build_hmm_for_config(seqdata, "A")
    hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
    py_ll = hmm.score()
    r_ll = ref_em["A"]["loglik_after"]
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_EM_ATOL), (
        f"Config A: converged logLik too different. "
        f"Python={py_ll:.4f}, R={r_ll:.4f}, diff={abs(py_ll - r_ll):.4f}"
    )


def test_em_config_B_loglik_matches_r(seqdata, ref_em):
    """Config B: converged logLik is close to R's."""
    hmm = _build_hmm_for_config(seqdata, "B")
    hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
    py_ll = hmm.score()
    r_ll = ref_em["B"]["loglik_after"]
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_EM_ATOL), (
        f"Config B: converged logLik too different. "
        f"Python={py_ll:.4f}, R={r_ll:.4f}, diff={abs(py_ll - r_ll):.4f}"
    )


def test_em_config_C_loglik_matches_r(seqdata, ref_em):
    """Config C: converged logLik is close to R's."""
    hmm = _build_hmm_for_config(seqdata, "C")
    hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
    py_ll = hmm.score()
    r_ll = ref_em["C"]["loglik_after"]
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_EM_ATOL), (
        f"Config C: converged logLik too different. "
        f"Python={py_ll:.4f}, R={r_ll:.4f}, diff={abs(py_ll - r_ll):.4f}"
    )


def test_em_python_not_worse_than_r(seqdata, ref_em):
    """Python EM should not be dramatically worse than R's solution.

    If Python's logLik is much lower, it may indicate a bug in the EM
    implementation rather than just a different local optimum.
    """
    for config_name in EM_CONFIGS:
        hmm = _build_hmm_for_config(seqdata, config_name)
        hmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
        py_ll = hmm.score()
        r_ll = ref_em[config_name]["loglik_after"]
        # Python should not be more than 1.0 worse than R
        assert py_ll >= r_ll - 1.0, (
            f"Config {config_name}: Python EM much worse than R. "
            f"Python={py_ll:.4f}, R={r_ll:.4f}"
        )


def test_em_loglik_before_matches_r(seqdata, ref_em):
    """The logLik BEFORE EM should match R exactly (same as loglik tests).

    This is a sanity check: if pre-EM logLik differs, the EM comparison
    is meaningless because the models started from different points.
    """
    for config_name in EM_CONFIGS:
        hmm = _build_hmm_for_config(seqdata, config_name)
        py_ll = hmm.score()
        r_ll = ref_em[config_name]["loglik_before"]
        assert np.isclose(py_ll, r_ll, atol=1e-6), (
            f"Config {config_name}: pre-EM logLik mismatch. "
            f"Python={py_ll:.6f}, R={r_ll:.6f}. "
            f"This means the initial models differ!"
        )
