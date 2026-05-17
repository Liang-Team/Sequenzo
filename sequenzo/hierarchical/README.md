# Hierarchical sequence analysis (`sequenzo.hierarchical`)

Analysis of **relational trajectories**: each sequence is one **level-1 × level-2** pair  
(e.g. region × technology, country × product, drug × indication) observed over time.

**Core questions**

1. **Decomposition:** Where does similarity among pair trajectories come from — **level-1**, **level-2**, or a **specific pair**?
2. **Typology:** What **types** of pair-level trajectories exist at scale (inactive, late entry, volatile, …)?

The module answers both in layers. `sequenzo.hierarchical.simulation` provides controlled experiments for recovery validation.

---

## Layered view (“千层蛋糕”)

| Layer | Question | Main outputs | Code |
|------:|----------|--------------|------|
| **1** | What do trajectories look like? | Long panel → pair sequences | `make_relational_sequences()`, `RelationalSequenceData` |
| **2** | How different are pairs? | Pairwise distance matrix | `compute_relational_distance_matrix()` |
| **3** | Marginal structure by each level? | Level-1 / level-2 **pseudo-R²** | `sequence_discrepancy_by_level()`, `plot_marginal_pseudo_r2()` |
| **4** | Joint vs pair-specific structure? | **Joint share** + **residual share** | `hierarchical_sequence_discrepancy()`, `plot_joint_residual_shares()` |
| **5** | Common **trajectory types** at scale? | `PairTypologyResult` | `cluster_pair_trajectories(algorithm="clara")` |
| **6** | Which **pairs** are exceptional? | Residuals + low representativeness | `compute_pair_residuals()`, typology `representativeness` |
| **7** | End-to-end | One pipeline | `run_hierarchical_sequence_analysis()` |

**Type-III additive decomposition**

- `level_1_share` and `level_2_share` are **partial / marginal** — not a partition.
- Only **`joint_share + residual_share = 1`** is a variance-style split.

**Typology vs decomposition**

| | Typology | Decomposition |
|---|----------|----------------|
| Asks | What types of pair trajectories exist? | Where does similarity come from? |
| Scalable tool | CLARA (`algorithm="clara"`) | Stratified sampling (`sample_pairwise_distances`) |
| Result object | `PairTypologyResult` | `HierarchicalDecompositionResult` |

---

## Package layout

```text
sequenzo/hierarchical/
  data.py, representation.py, distances.py   # core objects
  decomposition/
    marginal.py     # marginal pseudo-R², additive joint model
    crossed.py      # experimental crossed / interaction decomposition
    sampling.py     # large-n pairwise sampling (no full n×n matrix)
  clustering/
    pam.py          # full-matrix K-medoids (pair / level-1 / level-2)
    clara.py        # scalable pair-level CLARA typology
    typology.py     # cluster_pair_trajectories() user API
    results.py      # PairTypologyResult, HierarchicalClusterResult
    aggregate.py    # block-mean aggregation to higher levels
  residuals.py, profiles.py, results.py, visualization.py
  simulation/       # validation suite
```

Import from the top level:

```python
from sequenzo.hierarchical import (
    run_hierarchical_sequence_analysis,
    cluster_pair_trajectories,
    sample_pairwise_distances,
    hierarchical_sequence_discrepancy,
)
```

Or from subpackages:

```python
from sequenzo.hierarchical.decomposition import additive_sequence_discrepancy
from sequenzo.hierarchical.clustering import PairTypologyResult
```

---

## Recommended workflow (analysis)

```python
from sequenzo.hierarchical import run_hierarchical_sequence_analysis

result = run_hierarchical_sequence_analysis(
    data,
    level_1_col="region_id",
    level_2_col="technology_id",
    time_col="year",
    state_col="state",
    distance_method="HAM",
    n_perm=999,
)

print(result.summary())
result.plot_marginal_pseudo_r2()
result.plot_joint_residual_shares()
result.plot_outliers()
```

**Scalable pair-level typology (large n)**

```python
from sequenzo.hierarchical import make_relational_sequences, cluster_pair_trajectories

sequences = make_relational_sequences(data, "region_id", "technology_id", "year", "state")
typology = cluster_pair_trajectories(
    sequences,
    k=8,
    algorithm="clara",
    sample_size=5000,
    n_iterations=100,
    distance_method="HAM",
)
print(typology.to_dataframe())
```

Pipeline with typology (no full `n×n` matrix):

```python
result = run_hierarchical_sequence_analysis(
    data, ...,
    analysis_mode="typology_only",
    cluster_k=8,
    typology_algorithm="clara",
    typology_n_iterations=100,
)
# result.pair_typology — result.distance_matrix is None
```

Scalable decomposition (stratified pair sampling):

```python
result = run_hierarchical_sequence_analysis(
    data, ...,
    analysis_mode="sampled",
    n_same_level_1=200_000,
    n_same_level_2=200_000,
    n_baseline_pairs=400_000,
)
# result.distance_matrix is None; result.sampled_distances is set
```

**Sampled mode scope:** reports contrast-based structural decomposition only. Level profiles, pair residuals, and outlier tables need a full distance matrix (`analysis_mode="exact"` or `compute_full_distance=True`).

---

## Two-stage narrative (paper)

1. **Stage A — typology:** `cluster_pair_trajectories` → common pair trajectory types.  
2. **Stage B — decomposition:** `hierarchical_sequence_discrepancy` → level-1 / level-2 / residual structure.  
3. **Linkage:** cross-tabulate types with regions/technologies; flag pairs with high \|residual\| and low representativeness.

---

## Simulation validation (multilevel repo)

Recovery experiments and batch runners live in the **multilevel** paper repo
(`simulations/01_simulation_core_code/`). Pytest for generators, recovery, and
legacy simulators: [`multilevel/tests/`](../../../multilevel/tests/).

Runnable simulation scripts: [`multilevel/simulations/`](../../../multilevel/simulations/README.md).

---

## Scalability

| Topic | Detail |
|-------|--------|
| Full matrix | ~**n² × 8 bytes**; warning at `DEFAULT_MAX_FULL_MATRIX_PAIRS = 8_000` |
| Decomposition at scale | `sample_pairwise_distances(..., sampling_unit="structural")` or `analysis_mode="sampled"` |
| Typology at scale | `cluster_pair_trajectories(algorithm="clara")` — subsample + medoids, no full n×n storage |
| Identical trajectories | `compress_identical_relational_sequences()` — pattern ids + weights before CLARA |

**Sampled decomposition is not exact ANOVA.** `analysis_mode="sampled"` and `hierarchical_sequence_discrepancy_from_sample()` report contrast-based marginal pseudo-R² and heuristic joint/residual proxies (`method="structural_sample"`). They preserve the *direction* of level-specific structure but are **not** the Gower Type-III additive decomposition from a full distance matrix.

```python
from sequenzo.hierarchical import sampling_scheme_description, sample_pairwise_distances

print(sampling_scheme_description("structural"))
sample = sample_pairwise_distances(
    sequences,
    sampling_unit="structural",
    n_same_level_1=200_000,
    n_same_level_2=200_000,
    n_baseline=400_000,
)
```

---

## Module map

| Path | Role |
|------|------|
| `data.py` | Long format → `RelationalSequenceData` |
| `distances.py` | Pair distance matrix |
| `decomposition/marginal.py` | Marginal pseudo-R², permutation tests |
| `decomposition/crossed.py` | Additive & experimental crossed decomposition |
| `decomposition/sampling.py` | Pair / sequence / **structural** subsampling |
| `compression.py` | Identical trajectory compression + weights |
| `clustering/pam.py` | Full-matrix K-medoids |
| `clustering/clara.py` | CLARA typology for pair trajectories |
| `clustering/typology.py` | `cluster_pair_trajectories()` |
| `residuals.py` | Pair-specific residuals |
| `profiles.py` | Level profile summaries |
| `results.py` | `run_hierarchical_sequence_analysis()` |
| `visualization.py` | Plots |
| `simulation/` | Validation suite |

---

## Tests

```bash
python3 -m pytest tests/hierarchical/ -v
```

---

## References in code

- Additive joint + residual: `AdditiveDecompositionResult` in `decomposition/crossed.py`
- Residual interpretation: `residuals.py`
- CLARA backend: `sequenzo.big_data.clara`, wrapped in `clustering/clara.py`
