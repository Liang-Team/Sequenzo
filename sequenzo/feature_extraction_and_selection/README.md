# Feature Extraction and Selection (FES)

This module implements an FES-inspired workflow (Bolano & Studer 2020; Unterlerchner et al. 2023)—not a byte-for-byte R reproduction. It combines spell-based feature extraction, optional Boruta selection, and optional exploratory modeling. See [Method notes](#method-notes) and [Why Python cannot match R Boruta defaults exactly](#why-python-cannot-match-r-boruta-defaults-exactly).

Primary references:

| Paper | Role |
|-------|------|
| Bolano & Studer (2020) | Duration / timing / sequencing feature extraction; Boruta + stability selection |
| Unterlerchner, Studer & Gomensoro (2023) | Educational pathways & income; `preset="unterlerchner2023"`; `clustassoc` validation |

---

## Entry points

- `extract_sequence_features()`: **feature extraction only**.
  - Returns duration / timing / sequencing feature matrices.
  - Does **not** run Boruta.
  - Does **not** fit a final predictive / explanatory model.

- `run_feature_extraction_and_selection_pipeline()`: **full workflow**.
  - Extracts features.
  - Runs feature selection (Boruta).
  - Optionally fits an exploratory final model (`fit_final_model=False` by default).
  - Supports residualization with controls.

- `clustassoc_like_typology_validation()`: validate whether a clustering solution accounts for a covariate–sequence association (WeightedCluster `clustassoc` analogue).

- `select_relevant_features()` / `interpret_selected_features()` / `cluster_correlated_features()`: Boruta selection and post-selection interpretation (hierarchical clustering on `1 - |corr|` per Unterlerchner et al. 2023).

### Method notes

| Topic | Behavior |
|-------|----------|
| **Timing** | `START_*` = spell entry; `END_*` = spell exit. Preset `unterlerchner2023` sets `timing_include_end=True` and `end_time_mode="exit_time"`. |
| **Bin width** | `timing_bin_width` is in the **same unit as** `seqdata.time` (not always “months”). Monthly grids: use `12.0`; yearly age labels: use `1.0`. |
| **`time_unit_hint`** | **Metadata only** unless a preset calls `suggest_timing_bin_width()` for you. Setting `time_unit_hint="year"` does **not** change bins by itself—set `timing_bin_width` explicitly. |
| **Duration** | Summed spell lengths in **sequence-position steps** (`duration_steps`), not necessarily calendar months. |
| **Sequencing** | Mined on **spell-state** events (DSS-style), not raw repeated panels. |
| **Boruta** | Runtime dependency `boruta` (BorutaPy); installed with `pip install sequenzo`. **Confirmed** features: `selected_*`; **tentative**: `tentative_*` (BorutaPy `support_weak_`). |
| **Final model** | **Off by default** (`fit_final_model=False`). Result fields `final_model_fitted` / `final_model_is_exploratory` are `False` unless you opt in. Papers often cluster correlated Boruta features before regression—use `cluster_correlated_features()`. |
| **Classification controls** | When `residualize_target_with_controls=True`, binomial GLM **deviance residuals** are used for selection (**binary outcomes only**). Multi-class: set `residualize_target_with_controls=False`. Outcomes are encoded `0 … K-1` via `pd.Categorical`. |
| **`problem_type="auto"`** | Numeric outcomes default to **regression**. For binary `0/1` health or status codes, pass `problem_type="classification"` if you want classification + binomial residualization. |

### Reproducible preset

For paper-style reproducibility:

- `preset="unterlerchner2023"` — monthly `time_unit_hint`, 12-unit bins, start/end timing, `exit_time` ends, `max_k=3`, `min_support=0.05`, residualization; **no auto final model**.

```python
from sequenzo.feature_extraction_and_selection import (
    run_feature_extraction_and_selection_pipeline,
)

result = run_feature_extraction_and_selection_pipeline(
    seqdata=seqdata,
    outcome=outcome,
    controls=controls,
    preset="unterlerchner2023",
)
```

---

## Usage

### Unterlerchner (2023) style: extract + Boruta, no automatic final regression

```python
from sequenzo import run_feature_extraction_and_selection_pipeline

result = run_feature_extraction_and_selection_pipeline(
    seqdata=seqdata,
    outcome=outcome,
    controls=controls,
    preset="unterlerchner2023",
)
# result["selected_feature_names"], result["X_selected"], ...
```

### Exploratory final model (optional; not the paper’s main interpretive step)

```python
result = run_feature_extraction_and_selection_pipeline(
    seqdata=seqdata,
    outcome=outcome,
    controls=controls,
    preset="unterlerchner2023",
    fit_final_model=True,
)
```

Papers usually **cluster correlated Boruta features** and pick representatives before a final regression—not all selected features at once.

### Redundant-feature interpretation (Unterlerchner et al. 2023)

```python
from sequenzo import cluster_correlated_features

clusters = cluster_correlated_features(
    result["X_selected"],
    result["selected_feature_names"],
    abs_corr_threshold=0.7,
)
```

### Timing bins and `seqdata.time` units

`timing_bin_width` is always in the **same unit as** `seqdata.time` labels—not “months” unless your grid is monthly.

| `seqdata.time` | Set |
|----------------|-----|
| Monthly position index (TREE: `1 … 172`) | `time_unit_hint="month"`, `timing_bin_width=12.0` (preset `unterlerchner2023`) |
| Age in years (`15, 16, 17, …`) | `timing_bin_width=1.0` for one-year bins—**do not use 12** |
| Custom grid | Set `timing_bin_width` explicitly; `suggest_timing_bin_width("month")` or `suggest_timing_bin_width("year")` as starting points |

`time_unit_hint` on `FeatureExtractionAndSelectionConfig` is stored in pipeline results for documentation; it does **not** auto-adjust `timing_bin_width` unless you use a preset that sets both (e.g. `unterlerchner2023`).

Example for **yearly** age-labelled sequences:

```python
from sequenzo.feature_extraction_and_selection import (
    extract_sequence_features,
    FeatureExtractionAndSelectionConfig,
)

features = extract_sequence_features(
    seqdata,
    timing_bin_width=1.0,
    time_unit_hint="year",
    timing_include_end=True,
    end_time_mode="exit_time",
)
```

`boruta` is a **runtime dependency** of Sequenzo (see `pyproject.toml`); `pip install sequenzo` installs it automatically.

---

## Parity checklist: what still differs from the papers?

After the method fixes in this module, gaps fall into a few **clear buckets**. For **feature extraction + clustassoc + paper workflow**, Sequenzo is aligned with Bolano & Studer (2020) and Unterlerchner et al. (2023) **except** where noted below.

### A. Unavoidable without changing the Python Boruta stack (“爱莫能助”)

These are **not** Sequenzo bugs; they come from R `Boruta` + **ranger** permutation importance vs PyPI `boruta` (BorutaPy) + **sklearn** Gini importance:

| Topic | Impact |
|-------|--------|
| Importance metric | R default: permutation (MDA); BorutaPy: Gini `feature_importances_` |
| Random forest engine | **ranger** vs **sklearn** → different trees and importances |
| Exact Confirmed feature set | Same `X`, `y` can yield different selected features |

See [Why Python cannot match R Boruta defaults exactly](#why-python-cannot-match-r-boruta-defaults-exactly). With `boruta` installed, Sequenzo sets `boruta_two_step=False` and `boruta_alpha=0.01` to match R’s testing path as far as BorutaPy allows—**Gini vs permutation remains**.

### B. BorutaPy tuning (optional)

| Topic | Note |
|-------|------|
| BorutaPy `n_estimators="auto"` vs R `ntree=500` | Tune via custom estimator in `select_relevant_features()` if needed |

### C. Documented out-of-scope (papers mention; not in this submodule)

| Topic | Paper | Status |
|-------|-------|--------|
| Stability selection (`stabs` / LASSO) | Bolano & Studer (2020) | **Not implemented** (complementary to Boruta) |
| Complexity index (`seqici`, etc.) | Bolano & Studer (2020) optional | **Not implemented** |
| Weighted frequent subsequence mining | Possible extension | **Not wired** until `EventSequenceData` accepts weights |

These are **optional or complementary** steps in the articles, not part of the core Unterlerchner (2023) Boruta path.

### D. Minor engineering differences (same class as other Sequenzo ↔ TraMineR ports)

| Topic | Note |
|-------|------|
| `seqefsub` / event mining | Sequenzo `event_sequences` vs TraMineR C++; same semantics, possible tiny numeric/support edge cases |
| R `seqpropclust` timing options | Paper-style timing bins are implemented explicitly in Sequenzo (spell START/END in equal-width bins), not by wrapping a single R `seqpropclust` option |
| Spell sequencing events | Spell-start DSS path; aligned with Bolano sequencing-on-distinct-states |

### E. Methodology we **do** match (post-fix)

| Step | Alignment |
|------|-----------|
| Duration | Total spell steps per state (`seqistatd`-style) |
| Timing | START/END spell events in equal-width bins; 2023 preset: end events + `exit_time` |
| Sequencing | Frequent spell-state subsequences (`seqefsub`-style) |
| Residualization | OLS/WLS (regression); binomial deviance residuals (classification) |
| clustassoc | `dissmfacw`-style pseudo-R² accounting |
| Final regression | Off by default; correlation clustering helper for interpretation |

**Bottom line:** For the **published FES pipeline (features → Boruta → interpret)**, remaining disagreement with R runs should come from **bucket A** (BorutaPy vs R Boruta)—not from missing packages, timing/duration/sequencing logic, or an internal fallback.

---

## Python module ↔ R / literature mapping

There is **no one-to-one TraMineR function** for the full FES pipeline. The closest R “feature extraction” entry point is **WeightedCluster**, not TraMineR or TraMineRextras.

### End-to-end workflow

| Python | R / packages | Notes |
|--------|--------------|-------|
| `run_feature_extraction_and_selection_pipeline()` | `WeightedCluster::seqpropclust(..., prop.only=TRUE)` + `Boruta::Boruta()` + `lm()` / `glm()` | Papers describe this as a scripted combination, not one function |
| `extract_sequence_features()` | `seqpropclust(..., prop.only=TRUE)` | Feature matrices only |
| `preset="unterlerchner2023"` | Unterlerchner et al. (2023) parameterization | Custom timing bins; not a single R preset object |
| `clustassoc_like_typology_validation()` | `WeightedCluster::clustassoc()` | Uses `TraMineR::dissmfacw()` internally in R |

Bolano & Studer (2020) R workflow sketch (Section 6 of the paper):

```r
library(WeightedCluster)
features <- seqpropclust(myseq,
                        properties = c("pattern", "agerange", "duration"),
                        prop.only = TRUE)
library(Boruta)
Boruta(residuals(regconfond) ~ ., data = features)
```

### File-level mapping

| Python file | Function | TraMineR / other R | Package |
|-------------|----------|-------------------|---------|
| `monthly_state_to_spells.py` | `extract_spells_with_times()` | `seqdss()`, `seqdur()`; `seqformat(..., to="SPELL")` | TraMineR |
| `duration_timing_feature_builders.py` | `build_duration_features()` | `seqistatd()`; `seqpropclust` `properties="duration"` | TraMineR; WeightedCluster |
| `duration_timing_feature_builders.py` | `build_timing_features()` | **No direct TraMineR equivalent**; Unterlerchner (2023) custom bins on spell start/end | Closest R ideas: `seqpropclust` `AFpattern` or hand-coded bins |
| `sequencing_feature_builders.py` | `build_sequencing_features()` | `seqecreate()` → `seqefsub()` → `seqeapplysub()`; `seqpropclust` `properties="pattern"` | TraMineR |
| `time_binning_utils.py` | bin helpers | No direct R counterpart | Sequenzo utility |
| `boruta_feature_selection.py` | Boruta step | `Boruta::Boruta()` | R package **Boruta** |
| `clustassoc_typology_validation.py` | typology validation | `WeightedCluster::clustassoc()` + `TraMineR::dissmfacw()` | WeightedCluster + TraMineR |

### Sequenzo dependencies used inside this submodule

| Sequenzo module | TraMineR analogue |
|---------------|-----------------|
| `sequenzo.prefix_tree.convert_seqdata_to_spells` | `seqdss` + `seqdur` |
| `sequenzo.event_sequences.find_frequent_subsequences` | `seqefsub` |
| `sequenzo.event_sequences.count_subsequence_occurrences` | `seqeapplysub` |
| `sequenzo.discrepancy_analysis.stats.multifactor_association.distance_multifactor_anova` | `dissmfacw` |

### TraMineRextras

**No direct FES counterpart.** Loosely related but different goals:

| TraMineRextras | Purpose | Relation to FES |
|----------------|---------|-----------------|
| `extractSubseq` (seqsamm) | Person-period data + subsequence typology for SHA | Different method (SHA), not Boruta/FES |
| HSPELL → STS helpers | Hierarchical spell format conversion | Data format only |

### Important differences vs R papers (when reproducing)

1. **Timing**: Python uses spell start/end times in equal-width bins (Unterlerchner 2023). The paper-style timing bins are implemented explicitly in Sequenzo rather than by wrapping a single R `seqpropclust` option.
2. **Sequencing events**: Python builds event sequences at **spell starts** on the DSS spell path; R `pattern` uses `seqecreate(..., tevent="state")`—same intent, minor implementation differences possible.
3. **Boruta**: See [Parity checklist](#parity-checklist-what-still-differs-from-the-papers)—only unavoidable gaps are the Python/R Boruta stack when `boruta` is installed.
4. **Stability selection (LASSO)**: Bolano & Studer (2020) complementary method via `stabs`; **not implemented** (out of scope for core pipeline).
5. **Complexity index**: Optional in Bolano & Studer (2020); **not implemented**.

---

## Boruta: R package vs Python options vs this module

FES papers use **Boruta** (Kursa & Rudnicki, 2010) for *all-relevant* feature selection. Sequenzo wraps **BorutaPy** (PyPI package `boruta`); R papers typically use the **Boruta** R package with **ranger**.

| Stack | Role |
|-------|------|
| R **Boruta** | Reference implementation in published FES workflows |
| PyPI **`boruta`** (BorutaPy) | Sequenzo runtime dependency |
| `boruta_feature_selection.py` | Thin wrapper around BorutaPy |

See [Why Python cannot match R Boruta defaults exactly](#why-python-cannot-match-r-boruta-defaults-exactly) for the parity limits.

### R `Boruta`

- **Algorithm**: Full Boruta wrapper—shadow attributes, iterative **Confirmed / Rejected / Tentative** decisions, binomial tests with Bonferroni adjustment (`pValue`, `mcAdj`), early elimination of rejected features.
- **Importance source**: Default `getImpRfZ` (random forest via **ranger**, Z-scores of mean decrease accuracy).
- **API**: Formula interface `Boruta(y ~ ., data=...)`, rich output (`finalDecision`, `ImpHistory`, `attStats`, plots).
- **Used in FES papers** as: `Boruta(residuals(confounder_model) ~ features)`.

### Python BorutaPy (Sequenzo runtime dependency)

Declared in `pyproject.toml` as a runtime dependency: `boruta>=0.3` (imports as `from boruta import BorutaPy`).

| Aspect | R `Boruta` | PyPI `boruta` (BorutaPy) |
|--------|------------|--------------------------|
| Algorithm family | Full Boruta with statistical tests | Port of Boruta around sklearn RF |
| Tentative / rejected handling | Yes, with `TentativeRoughFix` | Partial (`support_`, `support_weak_`) |
| Importance | ranger Z-scores (default) | sklearn `feature_importances_` |
| Threshold | `pbinom` + `pValue` + Bonferroni | `perc`, `alpha`, `max_iter` (Sequenzo defaults: `alpha=0.01`, `two_step=False`) |
| Default RF trees | via `getImp` / `...` | `n_estimators="auto"` in our wrapper |
| Formula / residuals | Native formula + user residualizes `y` | Matrix `X`, `y` only |

**Sequenzo strategy** (`boruta_feature_selection.py`): call **BorutaPy** only (`boruta` is a required dependency). Confirmed features use `boruta.support_`; tentative features use `boruta.support_weak_` (exposed as `tentative_*` in pipeline / `select_relevant_features()` results).

For paper reproduction, align `n_iter` / `perc` / RF size with the paper; expect residual differences vs R for reasons in [Why Python cannot match R Boruta defaults exactly](#why-python-cannot-match-r-boruta-defaults-exactly).

---

## Why Python cannot match R Boruta defaults exactly

**Short answer:** Sequenzo uses the same *Boruta idea* as R, but not the same *scoring engine*. So the list of selected features can legitimately differ even when `X` and `y` are identical. This is expected—not a bug in Sequenzo.

Unlike many TraMineR ports in this project (e.g. `seqdur`, `seqefsub`), Boruta has **no practical path to bitwise-identical R parity** with default settings. The gap is in how “feature importance” is defined and which random-forest library builds the trees.

### What stays the same (the Boruta “skeleton”)

Both R `Boruta` and Python `BorutaPy` follow Kursa & Rudnicki (2010):

1. Start with many candidate features and an outcome `y`.
2. Each iteration:
   - Build **shadow features** by shuffling real feature columns (a random baseline).
   - Train a random forest on real + shadow features.
   - Compare each real feature’s importance to the shadow features.
3. Repeat over many rounds.
4. Use statistical tests to label features as **Confirmed**, **Rejected**, or **Tentative**.
5. Drop rejected features from later rounds.

So yes: **the algorithm skeleton is the same**—shadow comparison, iterative confirm/reject, multiple-testing correction.

What differs is **how importance is measured**, **which RF implementation is used**, and **several default tuning choices** in BorutaPy.

### Blocker 1 — Permutation importance (R default) vs Gini importance (Python default)

Think of Boruta as a repeated exam: “Is this feature better than random noise?”

| | R `Boruta` default | Python `BorutaPy` default |
|--|-------------------|---------------------------|
| RF library | **ranger** | **scikit-learn** |
| Importance | **Permutation** (`getImpRfZ`): shuffle one feature, measure how much **prediction** worsens | **Gini / impurity** (`feature_importances_`): sum how much each feature **splits** the data in the trees |
| Question answered | “If I break this feature, does the model suffer?” | “If I use this feature to split, how much purity do I gain?” |

**Why that changes the result**

- A feature can be **useful for prediction** (high permutation score) but not the best splitter (moderate Gini score), or the opposite.
- Boruta decides feature-by-feature whether real importance beats the best shadow. Change the scoring rule → change who wins → change the final Confirmed set.

R’s default aligns closely with Boruta’s shadow logic (both are about “what happens when values are randomized”). Python’s default is faster and fits the sklearn ecosystem, but it is **not the same measurement**.

BorutaPy’s own README states that importances come from **Gini impurity**, not R’s permutation (MDA) importance.

### Blocker 2 — ranger vs sklearn (even with the same importance *name*)

Even if you force R to use Gini (`getImp = getImpRfGini`) and compare to Python, results may still differ because:

- Tree building, tie-breaking, and subsampling differ between **ranger** and **sklearn**.
- Default tree counts differ (R often uses `ntree = 500`; Sequenzo may use `n_estimators = "auto"` or other values).
- Random seeds do not propagate identically across libraries and `n_jobs` settings.

So the second blocker is not only “permutation vs Gini” on paper—it is **two different random-forest engines producing different importance vectors**.

### Why BorutaPy was implemented differently (not just a literal R port)

`boruta_py` (PyPI package `boruta`) was written for the **sklearn workflow**, not as a line-by-line translation of R `Boruta`:

| Design choice in BorutaPy | R `Boruta` default | Why the Python author did it |
|---------------------------|-------------------|------------------------------|
| `feature_importances_` (Gini) | `getImpRfZ` (permutation) | Native in sklearn; much faster than per-round permutation |
| `two_step = TRUE` | Bonferroni only (`mcAdj = TRUE`) | Author considered strict Bonferroni too harsh for some applications |
| `perc` (percentile of shadows) | `max(shadow)` | Finer control when many features remain |
| `n_estimators = "auto"` | `ntree = 500` | Adapt tree count to feature dimension |
| No tests until iteration 5 | Tests from iteration 1 | Implementation detail in BorutaPy |
| Auto rough-fix for Tentative | Manual `TentativeRoughFix()` | Convenience at end of `fit()` |

BorutaPy documents that **`two_step = FALSE` and `perc = 100`** bring the *testing logic* closer to R—but **importance is still Gini**, so defaults still will not match R defaults.

### Three layers—what matches and what does not

| Layer | Match? | What it means |
|-------|--------|----------------|
| **Idea** | Yes | All-relevant selection via shadow features and iterative testing |
| **Procedure** | Mostly | Same loop structure; different defaults (`pValue` / `alpha`, `maxRuns`, `two_step`, when tests start) |
| **Numbers** | No (by default) | Different importance definition + different RF implementation → different Confirmed sets |

### What this means if you use Sequenzo

| Your goal | Realistic expectation |
|-----------|----------------------|
| Reproduce FES **methodology** (extract features → Boruta → model) | Supported; this is what the module is for |
| Get the **exact same features** as R `Boruta()` with default settings | **Not guaranteed**; often **not possible** without calling R |
| Compare Python vs R on a fixed `X`, `y` | Use a **set comparison** (e.g. overlap of Confirmed features), not exact equality, unless R also uses Gini and aligned hyperparameters |
| TraMineR-style automatic R reference tests | Feasible only for **small fixed matrices** and **documented tolerances**—not the same bar as `seqdur`-level numeric parity |

Sequenzo uses **BorutaPy** via the required `boruta` package (`pyproject.toml`).

### Getting *closer* to R (still not identical)

To narrow the gap when comparing against R **on purpose** (e.g. validation studies):

**Python (BorutaPy)** — closer testing logic, still Gini importance:

```python
BorutaPy(
    estimator,  # e.g. RandomForestRegressor(n_estimators=500, n_jobs=1)
    n_estimators=500,
    max_iter=100,
    perc=100,
    alpha=0.01,       # R default pValue = 0.01
    two_step=False,   # R-style Bonferroni path in BorutaPy
    random_state=42,
)
```

**R** — if comparing against Gini rather than R’s default permutation:

```r
Boruta(y ~ ., data = df, getImp = getImpRfGini, pValue = 0.01, maxRuns = 100, ntree = 500)
```

For **true** R-default parity (permutation + ranger), you would need either an R call from Python (e.g. `rpy2`) or a custom Python port that recomputes permutation importance every Boruta iteration—the current `boruta` PyPI package does not do that.

---

## Module layout

```
feature_extraction_and_selection/
├── feature_extraction.py              # extract_sequence_features()
├── feature_extraction_and_selection_pipeline.py
├── monthly_state_to_spells.py
├── duration_timing_feature_builders.py
├── sequencing_feature_builders.py
├── time_binning_utils.py
├── boruta_feature_selection.py
├── selection.py
├── interpretation.py
├── clustassoc_typology_validation.py
└── README.md
```

---

## Related code outside this folder

| Location | Relevance |
|----------|-----------|
| **TraMineR** (CRAN) | `seqdss`, `seqdur`, `seqistatd`, `seqecreate`, `seqefsub`, `dissmfacw` |
| **TraMineRextras** (CRAN) | No FES pipeline; SHA / format helpers only |
| **WeightedCluster** (CRAN) | `seqpropclust()` — closest R feature-extraction bundle; `clustassoc()` for typology validation |
| **Boruta** (CRAN) | R Boruta reference used in FES papers |
| PyPI **`boruta`** | BorutaPy — Sequenzo runtime dependency |
| `Tutorials/feature_extraction_and_selection/` | Usage examples including `unterlerchner2023` preset |

## References

Bolano, D., & Studer, M. (2020). The link between previous life trajectories and a later life outcome: A feature selection approach.

Unterlerchner, L., Studer, M., & Gomensoro, A. (2023). Back to the features. Investigating the relationship between educational pathways and income using sequence analysis and feature extraction and selection approach. Swiss journal of sociology, 49(2), 417-446.