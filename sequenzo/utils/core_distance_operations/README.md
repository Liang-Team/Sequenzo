# Core Distance Operations

`sequenzo.utils.core_distance_operations` contains low-level, reusable distance
primitives shared by multiple high-level modules.

## Why this module exists

Some operations are mathematically common across methods such as discrepancy
analysis and clustering (for example weighted inertia contribution on distance
matrices). Those operations are performance-critical and implemented in the
compiled extension.

This module gives these primitives a clear semantic home under `utils`, instead
of tying them conceptually to a specific analysis family.

## Current API

- `weighted_inertia_contrib(distance_matrix, indices, weights)`
  - TraMineR-style weighted inertia contribution:
    `sum_j(w_j * d_ij) / sum_j(w_j)` over selected indices.
  - Backed by `sequenzo.utils.core_distance_operations.core_distance_c_code`.

## Relationship with other modules

- `discrepancy_analysis`: uses these primitives for `dissvar`/`dissassoc`
  calculations.
- `clustering`: independent high-level clustering module. It no longer hosts
  these core distance-operation bindings.
- `utils`: exposes shared building blocks so additional future modules can
  reuse them without semantic coupling to clustering.

