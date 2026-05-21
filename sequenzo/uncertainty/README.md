# `sequenzo.uncertainty`

Assess how sensitive sequence-analysis results are to **measurement uncertainty**.

This package is the public entry point for uncertainty workflows in Sequenzo. It is
designed to grow over time; each type of uncertainty can add its own functions while
sharing the same `get_*` / `plot_*` naming style.

## Currently available: timing uncertainty

The first implementation follows Ritschard & Liao (2026) and the R package
**MCseqReplic**: Monte Carlo simulation of **transition timing errors** (when state
changes are reported too early or too late). See
`developer/important-literature/2026-ritschard-liao-assessing-the-impact-of-timing-errors-in-sequence-analysis.md`.

Typical workflow:

1. **`get_timing_perturbed_sequences`** — build replicated sequence datasets with
   random timing shifts (`keep.dss`, `indep`, or `relative` models).
2. **`get_distance_matrices_per_replicate`** — distance matrix for each replicate
   (optional; can be faster with unique-sequence aggregation).
3. **`get_distance_matrix_stability`** — mean and standard deviation of distances
   across replicate sets (when you already have a list of distance matrices).
4. **`get_distance_timing_uncertainty`** — pairwise MC mean and standard error of
   distances (R `seqdistMCSE`; supports `n_jobs` for parallel runs).
5. **`print_distance_uncertainty`** / **`summarize_distance_uncertainty`** — inspect
   results in the console.
6. **`plot_distance_uncertainty_heatmap`** — heatmap of an uncertainty matrix
   (e.g. `diss.z` ratios).

### Quick example

```python
from sequenzo import SequenceData
from sequenzo.uncertainty import (
    get_timing_perturbed_sequences,
    get_distance_timing_uncertainty,
    print_distance_uncertainty,
    summarize_distance_uncertainty,
    plot_distance_uncertainty_heatmap,
)

seq = SequenceData(...)  # your data

alt = get_timing_perturbed_sequences(seq, J=1, R=10, include_obs=True)

result = get_distance_timing_uncertainty(
    seq, method="LCS", J=1, R=50, n_jobs=-1, rng=25,
)

print_distance_uncertainty(result)
summarize_distance_uncertainty(result)
plot_distance_uncertainty_heatmap(result, which="diss_z")
```

### Parameters worth knowing

| Parameter | Meaning |
|-----------|---------|
| `J` | Max timing shift (integer) or odd-length probability vector over shifts |
| `R` | Number of Monte Carlo replicates |
| `model` | `"keep.dss"` (default), `"indep"`, or `"relative"` |
| `random_engine` | `"numpy"` (default) or `"r"` for R `set.seed` parity (`n_jobs` must be 1) |
| `n_jobs` | Parallel workers for `get_distance_timing_uncertainty` (`-1` = all cores) |

### R parity

Use `random_engine="r"` when you need to match R **MCseqReplic** draws exactly.
Reference tests are under `tests/mc_replic/`.

## Planned extensions (not implemented yet)

- State / token uncertainty in sequences  
- Sampling uncertainty wrappers (complements existing clustering validation)  
- Additional `plot_*` helpers (MDS stability, cluster stability under timing error)

## Public API map

| User-facing name | Role |
|------------------|------|
| `get_timing_perturbed_sequences` | MC-replicated `SequenceData` list |
| `get_timing_error_distribution` | Poisson-based timing error probabilities (`MCpj`) |
| `get_distance_matrices_per_replicate` | List of distance matrices per MC set |
| `get_distance_matrix_stability` | Mean / SD of distances across MC sets |
| `get_distance_timing_uncertainty` | Pairwise MC distance mean & SE (`seqdistMCSE`) |
| `print_distance_uncertainty` | Pretty-print `DistMCResult` |
| `summarize_distance_uncertainty` | Weighted five-number summary |
| `plot_distance_uncertainty_heatmap` | Heatmap of an uncertainty matrix |

Advanced helpers (`get_distance_matrices_unique`, cluster / MDS correlation under MC,
spell-level duration models) are exported from `sequenzo.uncertainty` but documented
as secondary; see module source under `sequenzo/uncertainty/`.
