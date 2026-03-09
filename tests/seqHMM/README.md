# seqHMM Test Suite

Consistency tests for `sequenzo.seqhmm` against the R [`seqHMM`](https://github.com/helske/seqHMM) package.

**191 tests** across **12 test files** and **10 test directories**, covering EM fitting, Viterbi decoding, posterior probabilities, model comparison, bootstrap, simulation, MHMM, NHMM, and a full replication of the examples from the seqHMM JSS paper.

## Quick Start

```bash
# Run all seqHMM tests
conda activate seq
python -m pytest tests/seqHMM/ -v

# Run a single module
python -m pytest tests/seqHMM/EM/ -v

# Run with timing
python -m pytest tests/seqHMM/ -v --durations=10
```

## Prerequisites

**Python dependencies:**

```bash
pip install hmmlearn scipy numpy pandas pytest
```

**R reference values** (already included as CSV files; only re-run if modifying test data):

```bash
# Example for one module
cd tests/seqHMM/EM
Rscript seqhmm_reference_em.R .
```

Each test directory contains an R script (`seqhmm_reference_*.R`) that generates reference CSV files (`ref_*.csv`). These CSVs are committed to the repository so R is not required to run the tests.


## Test Architecture

Every test module follows the same two-part structure:

**Part 0 — Sanity checks (no R needed).** These tests verify that Python functions run without error, return correct types and shapes, and satisfy basic mathematical properties (probabilities sum to 1, log-likelihoods are finite and negative, etc.).

**Part 1 — Cross-language consistency (needs reference CSVs).** These tests compare Python outputs against pre-computed R `seqHMM` reference values loaded from CSV files. If the CSVs are missing, the tests attempt to generate them by calling `Rscript`; if R is unavailable, they are skipped.


## Test Directories

### `EM/` — Expectation-Maximization (12 tests)

Tests that `fit_model()` (EM algorithm) converges correctly and produces log-likelihoods matching R's `fit_model()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_em.py` | 12 | EM convergence, parameter validity, determinism, logLik vs R across 3 configurations |
| `seqhmm_reference_em.R` | — | Generates `ref_em_A/B/C.csv` |


### `loglik_AIC_BIC/` — Log-Likelihood, AIC, BIC (19 tests)

Tests for `score()`, `aic()`, `bic()`, and `n_parameters()` against R's `logLik()`, `AIC()`, `BIC()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_loglik.py` | 19 | logLik across 4 configs, parameter counting, AIC/BIC formulas, observation counting |
| `seqhmm_reference_loglik.R` | — | Generates `ref_loglik.csv`, `ref_hidden_paths_A.csv`, `ref_posterior_A.csv` |


### `viterbi/` — Viterbi Decoding (15 tests)

Tests that `predict()` (Viterbi algorithm) returns the same most-likely state paths as R's `hidden_paths()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_viterbi.py` | 15 | Path matching across 4 configs, log-probabilities, index ranges, determinism |
| `seqhmm_reference_viterbi.R` | — | Generates `ref_viterbi_A/B/C/D.csv` |


### `posterior probability/` — Posterior State Probabilities (13 tests)

Tests that `predict_proba()` matches R's `posterior_probs()` element-by-element.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_posterior.py` | 13 | Posterior shapes, sum-to-one, unit interval, argmax consistency, 4-config matching vs R |
| `seqhmm_reference_posterior.R` | — | Generates `ref_posterior_A/B/C/D.csv`, `ref_forward_backward_A.csv` |


### `mhmm/` — Mixture Hidden Markov Model (20 tests)

Tests for `build_mhmm()` and `MHMM.fit()` (mixture of HMMs for clustering sequences).

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_mhmm.py` | 20 | Build/fit, cluster probs, responsibilities, predict_cluster, EM logLik vs R |
| `seqhmm_reference_mhmm.R` | — | Generates `ref_mhmm_loglik.csv`, `ref_mhmm_em.csv`, `ref_mhmm_cluster.csv`, `ref_mhmm_hidden_paths.csv` |


### `nhmm/` — Non-Homogeneous HMM (38 tests)

Tests for `build_nhmm()`, `fit_nhmm()`, gradient computation, and NHMM simulation.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_nhmm.py` | 11 | Build/fit, covariate dimensions, eta shapes, logLik vs R |
| `test_seqhmm_gradient_nhmm.py` | 10 | Gradient sanity (no NaN/Inf, correct length), finite-difference verification for η_π, η_A, η_B |
| `test_seqhmm_simulate_nhmm.py` | 17 | Simulate from NHMM: output shapes, alphabet validity, determinism, covariate formula support |
| `seqhmm_reference_nhmm.R` | — | Generates `ref_nhmm_fit.csv`, coefficient CSVs, `ref_nhmm_panel_data.csv` |


### `simulate/` — HMM & MHMM Simulation (14 tests)

Tests for `simulate_hmm()` and `simulate_mhmm()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_simulate.py` | 14 | Output dimensions, valid symbols, reproducibility, state/transition frequency convergence, symbol freq vs R |
| `seqhmm_reference_simulate.R` | — | Generates `ref_sim_hmm_sequences.csv`, `ref_sim_hmm_states.csv`, `ref_sim_hmm_stats.csv`, `ref_sim_mhmm_stats.csv` |


### `bootstrap/` — Bootstrap Confidence Intervals (13 tests)

Tests for `bootstrap_model()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_bootstrap.py` | 13 | Return structure, CI coverage, mean close to original, determinism with seed, consistency with R structure |
| `seqhmm_reference_bootstrap.R` | — | Generates `ref_bootstrap_stats.csv` |


### `advanced/` — Advanced Optimization & Model Comparison (23 tests)

Tests for `fit_model_advanced()`, formula parsing, and `compare_models()`.

| File | Tests | What it covers |
|------|-------|---------------|
| `test_seqhmm_advanced.py` | 23 | EM-only, EM+local, EM+global, random restarts, formula parsing, AIC/BIC comparison vs R |
| `seqhmm_reference_advanced.R` | — | Generates `ref_advanced_fit.csv`, `ref_advanced_comparison.csv`, `ref_advanced_formula.csv`, `ref_advanced_trim.csv` |


### `helske_example/` — Helske (2019) JSS Paper Replication (24 tests)

A systematic replication of the examples from Section 4 of:

> **Helske, S., & Helske, J. (2019).** Mixture Hidden Markov Models for Sequence Data: The seqHMM Package in R. *Journal of Statistical Software, 88*(3), 1–32. [https://doi.org/10.18637/jss.v088.i03](https://doi.org/10.18637/jss.v088.i03)

This replication uses the `biofam` life-course dataset (2000 sequences, ages 15–30) from the TraMineR package and the `biofam3c` three-channel variant from seqHMM. The tests mirror the paper's structure section by section:

| Paper Section | Test Class | What it covers |
|---------------|-----------|---------------|
| §4.1 Sequence data | `TestSection41_SequenceData` (6 tests) | Load biofam single-channel (8 states) and 3-channel (married/children/left), verify shapes and alphabets |
| §4.2 Single-channel HMM | `TestSection42A_SingleChannelHMM` (4 tests) | `build_hmm` + `fit_model` with Helske's exact initial parameters (sc_init, sc_trans, seqstatf-based sc_emiss), logLik vs R |
| §4.2 Multi-channel HMM | `TestSection42B_MultiChannelHMM` (5 tests) | `build_hmm` on `List[SequenceData]`, multichannel EM fitting, parameter validity |
| §4.3 Mixture HMM | `TestSection43_MHMM` (7 tests) | `build_mhmm` + `MHMM.fit()`, cluster probs, responsibilities, parameter validity |
| §4.4/4.5 Visualization | `TestSection44_45_Visualization` (2 tests) | Smoke test that `plot_hmm` and `plot_mhmm` are importable |

**Data files** (exported from R via `helske_export_data.R`):

| File | Description |
|------|-------------|
| `biofam_seq.csv` | Single-channel biofam sequences (2000 × 16 time points, 8 states) |
| `biofam3c_married.csv` | Marriage channel (single/married/divorced) |
| `biofam3c_children.csv` | Parenthood channel (childless/children) |
| `biofam3c_left.csv` | Residence channel (with parents/left home) |
| `ref_sc_emiss_init.csv` | R-computed initial emission matrix from `seqstatf()` |
| `ref_results.csv` | R reference values: `sc_loglik`, `mc_loglik`, `mc_bic` |

**Known limitations of this replication:**
- Multi-channel EM is implemented in pure Python (no C backend like R), so fitting tests use a 200-sequence subset to keep runtime reasonable.
- `build_mhmm()` only supports single-channel data, so §4.3 is tested as a single-channel surrogate (R uses 3-channel + covariates).
- Visualization (§4.4/4.5) is only tested for importability since `plot_hmm`/`plot_mhmm` use matplotlib (not igraph as in R).


## Source Code Bugs Found

During the development of this test suite, **5 bugs** were identified and fixed in `sequenzo.seqhmm`:

| # | File | Line(s) | Description | Severity |
|---|------|---------|-------------|----------|
| 1 | `mhmm.py` | 230–238 | **MHMM M-step ignores responsibilities.** The comment says "weighted fitting" but the code does equal-weight `self.clusters[k].fit()`, completely ignoring the E-step posteriors. Fixed with weighted Baum-Welch. | Critical |
| 2 | `bootstrap.py` | 157 | **`observations.values.columns` crashes.** `to_dataframe()` returns integer-coded values but `SequenceData()` expects string labels. Fixed by mapping integers back to labels via `.map()`. | High |
| 3 | `mhmm.py` | `__init__` | **MHMM missing `sequence_lengths` attribute.** Both HMM and NHMM define it, but MHMM does not, causing `bic(mhmm)` / `aic(mhmm)` to crash with `AttributeError`. | High |
| 4 | `build_hmm.py` | 96 | **`build_hmm()` crashes on `List[SequenceData]`.** The type signature declares `Union[SequenceData, List[SequenceData]]` but the implementation calls `observations.alphabet` without checking for list input. | High |
| 5 | `multichannel_em.py` | 184 | **Transition probability rows don't sum to 1.** `gamma_sum` accumulates over all time steps (t=0..T-1) but `xi_sum` only covers t=0..T-2, making the denominator too large. | Medium |


## File Summary

```
tests/seqHMM/
├── README.md                          ← This file
├── EM/
│   ├── test_seqhmm_em.py             (12 tests)
│   ├── seqhmm_reference_em.R
│   └── ref_em_A/B/C.csv
├── loglik_AIC_BIC/
│   ├── test_seqhmm_loglik.py         (19 tests)
│   ├── seqhmm_reference_loglik.R
│   └── ref_loglik.csv, ref_hidden_paths_A.csv, ref_posterior_A.csv
├── viterbi/
│   ├── test_seqhmm_viterbi.py        (15 tests)
│   ├── seqhmm_reference_viterbi.R
│   └── ref_viterbi_A/B/C/D.csv
├── posterior probability/
│   ├── test_seqhmm_posterior.py       (13 tests)
│   ├── seqhmm_reference_posterior.R
│   └── ref_posterior_A/B/C/D.csv, ref_forward_backward_A.csv
├── mhmm/
│   ├── test_seqhmm_mhmm.py           (20 tests)
│   ├── seqhmm_reference_mhmm.R
│   └── ref_mhmm_loglik/em/cluster/hidden_paths.csv
├── nhmm/
│   ├── test_seqhmm_nhmm.py           (11 tests)
│   ├── test_seqhmm_gradient_nhmm.py  (10 tests)
│   ├── test_seqhmm_simulate_nhmm.py  (17 tests)
│   ├── seqhmm_reference_nhmm.R
│   └── ref_nhmm_*.csv
├── simulate/
│   ├── test_seqhmm_simulate.py       (14 tests)
│   ├── seqhmm_reference_simulate.R
│   └── ref_sim_*.csv
├── bootstrap/
│   ├── test_seqhmm_bootstrap.py      (13 tests)
│   ├── seqhmm_reference_bootstrap.R
│   └── ref_bootstrap_stats.csv
├── advanced/
│   ├── test_seqhmm_advanced.py       (23 tests)
│   ├── seqhmm_reference_advanced.R
│   └── ref_advanced_*.csv
└── helske_example/
    ├── test_helske_replication.py     (24 tests)
    ├── helske_export_data.R
    ├── biofam_seq.csv
    ├── biofam3c_married/children/left.csv
    └── ref_results.csv, ref_sc_emiss_init.csv
```

## References

- **seqHMM R package:** Helske, S., & Helske, J. (2019). Mixture Hidden Markov Models for Sequence Data: The seqHMM Package in R. *Journal of Statistical Software, 88*(3), 1–32. [https://doi.org/10.18637/jss.v088.i03](https://doi.org/10.18637/jss.v088.i03)
- **seqHMM GitHub:** [https://github.com/helske/seqHMM](https://github.com/helske/seqHMM)
- **TraMineR (biofam data):** Gabadinho, A., Ritschard, G., Müller, N. S., & Studer, M. (2011). Analyzing and Visualizing State Sequences in R with TraMineR. *Journal of Statistical Software, 40*(4), 1–37.
- **hmmlearn:** [https://github.com/hmmlearn/hmmlearn](https://github.com/hmmlearn/hmmlearn)

---
Last Updated: March 2026
Author: Yapeng Wei
