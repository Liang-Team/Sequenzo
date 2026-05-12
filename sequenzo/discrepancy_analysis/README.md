# Discrepancy Analysis

This package implements discrepancy analysis for sequence distances and
covariates: overall discrepancy, pseudo-ANOVA, permutation tests, multifactor
models, regression trees, and window-wise comparisons. It mirrors the
TraMineR `diss*` family and related workflows described in Studer et al.
(2011).

Distance matrices are computed elsewhere (`sequenzo.dissimilarity_measures`).
Low-level weighted inertia kernels live in
`sequenzo.utils.core_distance_operations`.

## How to import

Use the package root only:

```python
from sequenzo.discrepancy_analysis import (
    overall_discrepancy,
    single_factor_association,
    multifactor_association,
    distance_tree,
    compare_groups_across_positions,
)
```

Subfolders (`stats/`, `trees/`, `positionwise/`, `internal/`) exist for
maintainers. They are not a second public API surface.

## Public API vs TraMineR

| User goal | Sequenzo (public) | TraMineR |
| --- | --- | --- |
| Overall discrepancy | `overall_discrepancy` | `dissvar` |
| Single-factor group comparison | `single_factor_association` | `dissassoc` |
| Multifactor model (controlled effects) | `multifactor_association`, `distance_multifactor_anova` | `dissmfacw` |
| One factor at a time (marginal) | `marginal_factor_association` | repeated `dissassoc` |
| Individual marginality / gain | `individual_indicators` | TraMineRextras `dissindic` |
| Merge cluster labels | `merge_cluster_groups` | `dissmergegroups` |
| Distance-based tree | `distance_tree` | `disstree` |
| Sequence + covariate tree | `sequence_tree` | `seqtree` |
| Leaf rules / assignment | `get_classification_rules`, `assign_to_leaves` | `disstree` helpers |
| Position-wise differences | `compare_groups_across_positions` | `seqdiff` workflow |
| Permutation tests | `permutation_test`, `association_permutation_test` | `dissassocweighted.*` |

Do not use `marginal_factor_association` when you need TraMineR-style
`dissmfacw` (multifactor, Type II partial effects).

## Weighted permutation defaults

When `weights` is omitted, association and tree functions resolve
`weight_permutation` to `"none"`. When weights are supplied and the caller does
not override the mode, the default is `"replicate"`, matching TraMineR
`dissassoc()`, `disstree()`, and `seqtree()`. For survey or calibration
weights, pass `weight_permutation="diss"` explicitly, as recommended by Studer
et al. (2011). The `compare_groups_across_positions()` scan follows TraMineR
`seqdiff` and uses `"diss"` for weighted local association summaries.

## Layout

```text
discrepancy_analysis/
  __init__.py          # public exports
  stats/               # overall / single- / multifactor association, indicators, merge
  trees/               # distance tree, sequence tree, nodes, leaf helpers, plots
  positionwise/        # position-wise group comparisons
  internal/            # weighted inertia and permutation engines
```

### `stats/`

| File | TraMineR | Main symbols |
| --- | --- | --- |
| `overall_discrepancy.py` | `dissvar` | `overall_discrepancy` |
| `single_factor_association.py` | `dissassoc` | `single_factor_association` |
| `marginal_factor_association.py` | marginal `dissassoc` | `marginal_factor_association` |
| `multifactor_association.py` | `dissmfacw` | `multifactor_association`, `distance_multifactor_anova`, `gower_matrix` |
| `individual_indicators.py` | `dissindic` | `individual_indicators` |
| `merge_cluster_groups.py` | `dissmergegroups` | `merge_cluster_groups` |

### `trees/`

| File | TraMineR | Role |
| --- | --- | --- |
| `distance_tree.py` | `disstree` | fit distance tree |
| `sequence_tree.py` | `seqtree` | fit sequence tree |
| `tree_node.py` | tree objects | `DissTreeNode`, `DissTreeSplit` |
| `tree_leaf_helpers.py` | leaf utilities | membership, rules, assignment |
| `tree_visualization.py` | plotting | `plot_tree`, `print_tree`, `export_tree_to_dot` |

### `positionwise/`

| File | TraMineR | Role |
| --- | --- | --- |
| `compare_by_position.py` | `seqdiff` | compare groups along the time axis |

### `internal/`

| File | TraMineR | Role |
| --- | --- | --- |
| `weighted_inertia.py` | C inertia helpers | shared sum-of-squares / centering |
| `permutation_engine.py` | `TraMineR.permutation` | generic and tree-split permutations |
| `single_factor_permutation.py` | `dissassocweighted.*` | five-statistic `dissassoc` permutations |

## Dependencies

- `sequenzo.dissimilarity_measures` for distance matrices when not supplied
- `sequenzo.utils.core_distance_operations` for weighted inertia contributions

## Tests

TraMineR reference checks: `tests/discrepancy_analysis/`. Functional tests:
`tests/discrepancy_analysis/test_discrepancy_analysis_lsog.py`.
