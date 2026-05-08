# Discrepancy Analysis

This module implements discrepancy-analysis methods for sequence distances and
covariates (TraMineR-style `diss*` family and related workflows).

## Scope

`discrepancy_analysis` focuses on statistical association and decomposition over
distance matrices (pseudo-variance, pseudo-ANOVA, permutation testing, trees,
position-wise discrepancy).

It does **not** define generic distance-kernel primitives itself.

## Core dependency

For performance-critical low-level distance operations, this module uses:

- `sequenzo.utils.core_distance_operations`

In particular, weighted inertia contribution is delegated there, and that
utility is backed by `sequenzo.utils.core_distance_operations.core_distance_c_code`.

## Architecture note

- High-level method semantics: `discrepancy_analysis`
- Shared low-level distance kernels: `utils.core_distance_operations`
- Compiled backend implementation: `utils.core_distance_operations.core_distance_c_code`

This separation keeps method semantics clear while reusing a single optimized
backend across modules.

