"""
@Author  : Yapeng Wei
@File    : test_seqhmm_loglik.py
@Desc    :
Tests for HMM log-likelihood (and AIC/BIC) consistency between
sequenzo.seqhmm and R seqHMM (https://github.com/helske/seqHMM).

Methodology:
  - Synthetic data: 5 sequences, 8 time points, 3 observed states (A, B, C).
  - Four HMM parameter configurations (A, B, C, D) with manually specified
    initial_probs, transition_probs, and emission_probs (no randomness).
  - R seqHMM computes logLik(model), AIC, BIC as ground truth.
  - Python sequenzo.seqhmm computes HMM.score(), aic(), bic() and compares.

Test groups:
  Part 0: Sanity checks (no R needed)
    - build_hmm produces a valid HMM object
    - score() returns a finite negative number
    - score() is deterministic (same call twice = same result)
    - Config D (uniform) logLik equals analytical value 40*log(1/3)

  Part 1: logLik vs R seqHMM (needs ref_loglik.csv from R script)
    - Config A: 2 states, basic parameters
    - Config B: 3 states
    - Config C: 2 states, high self-transition (sticky)
    - Config D: 2 states, uniform emissions (null baseline)

  Part 2: AIC/BIC and parameter counting vs R seqHMM
    - n_parameters (df) matches R
    - n_observations (nobs) matches R
    - AIC matches R
    - BIC matches R

Run seqhmm_reference_loglik.R to generate ref_loglik.csv:
  Rscript seqhmm_reference_loglik.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm, aic, bic, compute_n_parameters, compute_n_observations


# ============================================================================
# Constants
# ============================================================================

# Observed states (alphabet) — order must match R script's alphabet = c("A","B","C")
STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)

# Sequence IDs
SEQ_IDS = ["s0", "s1", "s2", "s3", "s4"]

# Time column names
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]

# Directory of this test module (for R scripts and ref CSVs)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Synthetic test data (MUST be identical to R script)
# ============================================================================
# Seq s0: A A B B C A A B
# Seq s1: B C C A A B C C
# Seq s2: A B C A B C A B
# Seq s3: C C B A A A B C
# Seq s4: A A A B B B C C

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s0
    ["B", "C", "C", "A", "A", "B", "C", "C"],  # s1
    ["A", "B", "C", "A", "B", "C", "A", "B"],  # s2
    ["C", "C", "B", "A", "A", "A", "B", "C"],  # s3
    ["A", "A", "A", "B", "B", "B", "C", "C"],  # s4
]


# ============================================================================
# HMM parameter configurations (MUST be identical to R script)
# ============================================================================
# Each config is a dict with initial_probs, transition_probs, emission_probs.
# Emission matrix: rows = hidden states, columns = observed states (A, B, C).

CONFIGS = {
    "A": {
        "n_states": 2,
        "initial_probs": np.array([0.6, 0.4]),
        "transition_probs": np.array([
            [0.7, 0.3],
            [0.2, 0.8],
        ]),
        "emission_probs": np.array([
            [0.5, 0.3, 0.2],   # State 1: P(A)=0.5, P(B)=0.3, P(C)=0.2
            [0.1, 0.4, 0.5],   # State 2: P(A)=0.1, P(B)=0.4, P(C)=0.5
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
            [0.6, 0.3, 0.1],   # State 1
            [0.1, 0.6, 0.3],   # State 2
            [0.3, 0.1, 0.6],   # State 3
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
            [0.8, 0.15, 0.05],  # State 1: strongly emits A
            [0.05, 0.15, 0.8],  # State 2: strongly emits C
        ]),
    },
    "D": {
        "n_states": 2,
        "initial_probs": np.array([0.5, 0.5]),
        "transition_probs": np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        "emission_probs": np.array([
            [1.0 / 3, 1.0 / 3, 1.0 / 3],  # uniform
            [1.0 / 3, 1.0 / 3, 1.0 / 3],  # uniform
        ]),
    },
}


# ============================================================================
# Helper functions
# ============================================================================

def _make_test_seqdata():
    """Create SequenceData from the synthetic test data (same as R script)."""
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _build_hmm_for_config(seqdata, config_name):
    """Build an HMM with the specified config (no fitting, parameters set manually)."""
    cfg = CONFIGS[config_name]
    return build_hmm(
        observations=seqdata,
        n_states=cfg["n_states"],
        initial_probs=cfg["initial_probs"].copy(),
        transition_probs=cfg["transition_probs"].copy(),
        emission_probs=cfg["emission_probs"].copy(),
    )


def _run_r_reference(outdir, timeout=60):
    """Run seqhmm_reference_loglik.R to generate ref_loglik.csv in outdir.
    Returns True on success."""
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_loglik.R")
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, outdir],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return True


def _load_ref_loglik(ref_dir):
    """Load ref_loglik.csv and return as a dict keyed by config name.
    Each value is a dict with keys: loglik, df, nobs, aic, bic."""
    path = os.path.join(ref_dir, "ref_loglik.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        result[row["config"]] = {
            "loglik": float(row["loglik"]),
            "df": int(row["df"]),
            "nobs": int(row["nobs"]),
            "aic": float(row["aic"]),
            "bic": float(row["bic"]),
            "n_states": int(row["n_states"]),
        }
    return result


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    """SequenceData from synthetic test data (shared across all tests)."""
    return _make_test_seqdata()


@pytest.fixture(scope="module")
def ref_loglik():
    """Load R reference values for logLik/AIC/BIC.

    Priority:
      1. If ref_loglik.csv exists in THIS_DIR, use it directly.
      2. Otherwise, try running the R script to generate it.
      3. If R is not available, skip tests that need references.
    """
    # Check if ref file already exists
    path = os.path.join(THIS_DIR, "ref_loglik.csv")
    if os.path.isfile(path):
        refs = _load_ref_loglik(THIS_DIR)
        if refs is not None:
            return refs

    # Try running R script
    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = _load_ref_loglik(outdir)
        if refs is not None:
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_loglik.csv not found. "
        "Run: Rscript seqhmm_reference_loglik.R . "
        "to generate reference values."
    )


# ============================================================================
# Part 0: Sanity checks (no R reference needed)
# ============================================================================

class TestSanity:
    """Basic sanity checks that do not require R reference values."""

    def test_build_hmm_returns_hmm_object(self, seqdata):
        """build_hmm with valid params returns an HMM with correct attributes."""
        hmm = _build_hmm_for_config(seqdata, "A")
        assert hmm.n_states == 2
        assert hmm.n_sequences == 5
        assert hmm.initial_probs is not None
        assert hmm.transition_probs is not None
        assert hmm.emission_probs is not None

    def test_score_returns_finite_negative(self, seqdata):
        """HMM.score() returns a finite, negative log-likelihood."""
        hmm = _build_hmm_for_config(seqdata, "A")
        ll = hmm.score()
        assert np.isfinite(ll), f"score() returned non-finite value: {ll}"
        assert ll < 0, f"Log-likelihood should be negative, got {ll}"

    def test_score_is_deterministic(self, seqdata):
        """Same model + same data must always produce the same score."""
        hmm = _build_hmm_for_config(seqdata, "A")
        ll1 = hmm.score()
        ll2 = hmm.score()
        assert ll1 == ll2, f"score() not deterministic: {ll1} vs {ll2}"

    def test_score_varies_across_configs(self, seqdata):
        """Different parameter configs should yield different logLik values."""
        scores = {}
        for name in ["A", "B", "C", "D"]:
            hmm = _build_hmm_for_config(seqdata, name)
            scores[name] = hmm.score()
        # At least configs A, C, D should differ (B has 3 states, also different)
        assert scores["A"] != scores["C"], "Configs A and C should differ"
        assert scores["A"] != scores["D"], "Configs A and D should differ"

    def test_uniform_model_loglik_analytical(self, seqdata):
        """Config D (uniform emissions): logLik = n_obs * log(1/n_symbols).

        With uniform emission probabilities and uniform/symmetric transition
        and initial probabilities, the hidden states carry no information.
        Every observation has probability 1/3, so:
          logLik = 5 sequences * 8 timepoints * log(1/3) = 40 * log(1/3)
        """
        hmm = _build_hmm_for_config(seqdata, "D")
        ll = hmm.score()
        n_obs = 5 * 8  # 5 sequences, 8 time points each
        expected = n_obs * np.log(1.0 / N_SYMBOLS)
        assert np.isclose(ll, expected, atol=1e-10), (
            f"Uniform model logLik: got {ll}, expected {expected} "
            f"(= {n_obs} * log(1/{N_SYMBOLS}))"
        )

    def test_n_parameters_formula(self, seqdata):
        """Verify n_parameters formula: (S-1) + S*(S-1) + S*(M-1)."""
        for name, cfg in CONFIGS.items():
            hmm = _build_hmm_for_config(seqdata, name)
            n_params = compute_n_parameters(hmm)
            S = cfg["n_states"]
            M = N_SYMBOLS
            expected = (S - 1) + S * (S - 1) + S * (M - 1)
            assert n_params == expected, (
                f"Config {name}: n_parameters={n_params}, "
                f"expected {expected} for S={S}, M={M}"
            )

    def test_n_observations_count(self, seqdata):
        """n_observations = total observed time points (no missing values)."""
        hmm = _build_hmm_for_config(seqdata, "A")
        n_obs = compute_n_observations(hmm)
        expected = 5 * 8  # 5 sequences, 8 time points
        assert n_obs == expected, f"n_observations={n_obs}, expected {expected}"

    def test_aic_bic_formula(self, seqdata):
        """AIC = -2*logLik + 2*k, BIC = -2*logLik + log(n)*k."""
        hmm = _build_hmm_for_config(seqdata, "A")
        # We need to set log_likelihood for aic/bic to work
        ll = hmm.score()
        hmm.log_likelihood = ll
        k = compute_n_parameters(hmm)
        n = compute_n_observations(hmm)

        aic_val = aic(hmm)
        bic_val = bic(hmm)

        expected_aic = -2 * ll + 2 * k
        expected_bic = -2 * ll + np.log(n) * k

        assert np.isclose(aic_val, expected_aic, atol=1e-10), (
            f"AIC: got {aic_val}, expected {expected_aic}"
        )
        assert np.isclose(bic_val, expected_bic, atol=1e-10), (
            f"BIC: got {bic_val}, expected {expected_bic}"
        )


# ============================================================================
# Part 1: logLik vs R seqHMM
# ============================================================================
# Tolerance notes:
#   - R seqHMM uses custom C++ forward algorithm.
#   - Python (via hmmlearn) uses Cython forward algorithm in log-space.
#   - Both compute sum of log P(Y_i | model) over all sequences.
#   - Expected numerical difference: < 1e-10 for identical parameters.
#   - We use atol=1e-6 as a conservative safety margin.

LOGLIK_ATOL = 1e-6


def test_loglik_config_A_matches_seqhmm(seqdata, ref_loglik):
    """Config A (2 states, basic) logLik matches R seqHMM."""
    ref = ref_loglik["A"]
    hmm = _build_hmm_for_config(seqdata, "A")
    ll = hmm.score()
    assert np.isclose(ll, ref["loglik"], atol=LOGLIK_ATOL), (
        f"Config A logLik: sequenzo={ll}, R seqHMM={ref['loglik']}"
    )


def test_loglik_config_B_matches_seqhmm(seqdata, ref_loglik):
    """Config B (3 states) logLik matches R seqHMM."""
    ref = ref_loglik["B"]
    hmm = _build_hmm_for_config(seqdata, "B")
    ll = hmm.score()
    assert np.isclose(ll, ref["loglik"], atol=LOGLIK_ATOL), (
        f"Config B logLik: sequenzo={ll}, R seqHMM={ref['loglik']}"
    )


def test_loglik_config_C_matches_seqhmm(seqdata, ref_loglik):
    """Config C (2 states, sticky) logLik matches R seqHMM."""
    ref = ref_loglik["C"]
    hmm = _build_hmm_for_config(seqdata, "C")
    ll = hmm.score()
    assert np.isclose(ll, ref["loglik"], atol=LOGLIK_ATOL), (
        f"Config C logLik: sequenzo={ll}, R seqHMM={ref['loglik']}"
    )


def test_loglik_config_D_matches_seqhmm(seqdata, ref_loglik):
    """Config D (uniform, null) logLik matches R seqHMM."""
    ref = ref_loglik["D"]
    hmm = _build_hmm_for_config(seqdata, "D")
    ll = hmm.score()
    assert np.isclose(ll, ref["loglik"], atol=LOGLIK_ATOL), (
        f"Config D logLik: sequenzo={ll}, R seqHMM={ref['loglik']}"
    )


# ============================================================================
# Part 2: AIC, BIC, n_parameters, n_observations vs R seqHMM
# ============================================================================
# Note on degrees of freedom (df):
#   R seqHMM treats zero probabilities in the initial model as "structural
#   zeroes" and excludes them from the parameter count. Since none of our
#   test configs contain zero probabilities, df should match exactly.

def test_n_parameters_config_A_matches_seqhmm(seqdata, ref_loglik):
    """Config A: n_parameters (df) matches R seqHMM."""
    ref = ref_loglik["A"]
    hmm = _build_hmm_for_config(seqdata, "A")
    n_params = compute_n_parameters(hmm)
    assert n_params == ref["df"], (
        f"Config A n_parameters: sequenzo={n_params}, R seqHMM df={ref['df']}"
    )


def test_n_parameters_config_B_matches_seqhmm(seqdata, ref_loglik):
    """Config B: n_parameters (df) matches R seqHMM."""
    ref = ref_loglik["B"]
    hmm = _build_hmm_for_config(seqdata, "B")
    n_params = compute_n_parameters(hmm)
    assert n_params == ref["df"], (
        f"Config B n_parameters: sequenzo={n_params}, R seqHMM df={ref['df']}"
    )


def test_n_observations_matches_seqhmm(seqdata, ref_loglik):
    """n_observations (nobs) matches R seqHMM (all configs share same data)."""
    ref = ref_loglik["A"]  # nobs is the same for all configs (same data)
    hmm = _build_hmm_for_config(seqdata, "A")
    n_obs = compute_n_observations(hmm)
    assert n_obs == ref["nobs"], (
        f"n_observations: sequenzo={n_obs}, R seqHMM nobs={ref['nobs']}"
    )


def test_aic_config_A_matches_seqhmm(seqdata, ref_loglik):
    """Config A: AIC matches R seqHMM's AIC(model)."""
    ref = ref_loglik["A"]
    hmm = _build_hmm_for_config(seqdata, "A")
    hmm.log_likelihood = hmm.score()
    aic_val = aic(hmm)
    assert np.isclose(aic_val, ref["aic"], atol=LOGLIK_ATOL * 2), (
        f"Config A AIC: sequenzo={aic_val}, R seqHMM={ref['aic']}"
    )


def test_bic_config_A_matches_seqhmm(seqdata, ref_loglik):
    """Config A: BIC matches R seqHMM's BIC(model)."""
    ref = ref_loglik["A"]
    hmm = _build_hmm_for_config(seqdata, "A")
    hmm.log_likelihood = hmm.score()
    bic_val = bic(hmm)
    assert np.isclose(bic_val, ref["bic"], atol=LOGLIK_ATOL * 2), (
        f"Config A BIC: sequenzo={bic_val}, R seqHMM={ref['bic']}"
    )


def test_aic_config_B_matches_seqhmm(seqdata, ref_loglik):
    """Config B: AIC matches R seqHMM (3-state model, more parameters)."""
    ref = ref_loglik["B"]
    hmm = _build_hmm_for_config(seqdata, "B")
    hmm.log_likelihood = hmm.score()
    aic_val = aic(hmm)
    assert np.isclose(aic_val, ref["aic"], atol=LOGLIK_ATOL * 2), (
        f"Config B AIC: sequenzo={aic_val}, R seqHMM={ref['aic']}"
    )


def test_bic_config_C_matches_seqhmm(seqdata, ref_loglik):
    """Config C: BIC matches R seqHMM (sticky model)."""
    ref = ref_loglik["C"]
    hmm = _build_hmm_for_config(seqdata, "C")
    hmm.log_likelihood = hmm.score()
    bic_val = bic(hmm)
    assert np.isclose(bic_val, ref["bic"], atol=LOGLIK_ATOL * 2), (
        f"Config C BIC: sequenzo={bic_val}, R seqHMM={ref['bic']}"
    )
