# Clustering Module

This module provides clustering workflows and hosts Sequenzo's compiled C/C++
extension (`clustering_c_code`).

## What lives here

- Clustering algorithms and helpers (`KMedoids`, hierarchical clustering, etc.)
- Compiled backend (`clustering_c_code`) used for high-performance distance and
  cluster-related kernels.

## Important clarification

General distance-operation kernels have been migrated to
`sequenzo.utils.core_distance_operations` to keep semantic boundaries clear.
For example, weighted inertia contribution used by discrepancy analysis is now
provided by `core_distance_c_code` in that utility module.

## Layering

- `clustering`: algorithmic workflows + compiled backend host
- `utils.core_distance_operations`: shared low-level distance primitives
- `discrepancy_analysis` and other modules: high-level statistical methods that
  consume shared primitives

