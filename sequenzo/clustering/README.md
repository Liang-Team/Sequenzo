# Clustering module layout and TraMineR / WeightedCluster mapping

This package hosts clustering workflows and the compiled backend `clustering_c_code`.
File names follow the R ecosystem where a stable counterpart exists; otherwise they
describe the statistical workflow in plain language.

## Directory layout

```text
sequenzo/clustering/
  hierarchical_clustering.py       # hclust + clustrange-style quality
  k_medoids.py                     # weighted PAM / K-medoids
  sequences_to_variables/
    helske_regression_variables.py # Helske (2024) sequence -> regression covariates
    helpers.py                     # shared small helpers
  fuzzy/
    wfcmdd_fuzzy_clustering.py     # distance-based fuzzy C-medoids
    fuzzy_sequence_plots.py        # membership-weighted seq index plots
  validation/
    partition_quality.py           # fixed partitions + CQI table
    dissmfacw_factors.py           # multi-factor discrepancy association core
    bootstrap_cluster_range.py     # bootstrap CQI stability
    cluster_covariate_association.py
    rarcat_typology_regression.py
  utils/disscenter.py              # medoid lookup on distance matrices
  src/                             # C++ kernels (not part of the public API map)
```

## TraMineR and WeightedCluster mapping

| R package / function | Sequenzo module | Sequenzo symbol | Notes |
| --- | --- | --- | --- |
| `stats::hclust` | `hierarchical_clustering` | `Cluster` | Linkage on a distance matrix |
| WeightedCluster `as.clustrange` | `hierarchical_clustering` | `ClusterQuality`, `ClusterResults` | Multi-k partitions and CQI curves |
| WeightedCluster `wcClusterQuality` | `hierarchical_clustering` | `ClusterQuality` | PBC, ASW, CH, R2, HC, ... |
| WeightedCluster `wcKMedoids` | `k_medoids` | `KMedoids` | Weighted PAM / PAMonce |
| TraMineR `disscenter` | `utils.disscenter` | `disscentertrim` | Medoid indices from `diss` |
| WeightedCluster `as.clustrange.default` | `validation.partition_quality` | `cluster_range_from_partitions` | Known partition columns |
| TraMineR `dissmfacw` | `validation.dissmfacw_factors` | `dissmfacw_table` | Internal table for association |
| WeightedCluster `bootclustrange` | `validation.bootstrap_cluster_range` | `boot_cluster_range` | Bootstrap CQI summaries |
| WeightedCluster `clustassoc` | `validation.cluster_covariate_association` | `cluster_association` | Clustering vs covariate |
| WeightedCluster `rarcat` | `validation.rarcat_typology_regression` | `rarcat` | Robust typology AME |
| `cluster::fanny` (membership) | `sequences_to_variables.helske_regression_variables` | `fanny_membership` | Distance-based soft membership |
| WeightedCluster `wfcmdd` | `fuzzy.wfcmdd_fuzzy_clustering` | `wfcmdd` | FCMdd / NCdd / PCMdd |
| WeightedCluster `fuzzyseqplot` | `fuzzy.fuzzy_sequence_plots` | `fuzzy_sequence_plot` | Membership-weighted index plot |
| WeightedCluster `crispness` | `fuzzy.wfcmdd_fuzzy_clustering` | `crispness` | Partition sharpness |
| Helske representativeness | `sequences_to_variables.helske_regression_variables` | `representativeness_matrix` | Not a TraMineR export |
| Helske hard / soft / pseudoclass | `sequences_to_variables.helske_regression_variables` | `hard_classification_variables`, `soft_classification_variables`, `pseudoclass_regression` | Regression-ready typology covariates |

## Related code outside this folder

| R / topic | Sequenzo location |
| --- | --- |
| TraMineR `dissassoc`, `dissmfacw` (user-facing) | `sequenzo.discrepancy_analysis` |
| TraMineR `disstree`, `seqtree` | `sequenzo.discrepancy_analysis` |
| WeightedCluster `seqclararange`, `wcAggregateCases` | `sequenzo.big_data.clara` |

## Naming conventions for contributors

- Prefer workflow names (`bootstrap_cluster_range`) over opaque abbreviations in file names.
- Keep R function names on the Python **symbol** when the behavior is meant to match R one-to-one (`wfcmdd`, `rarcat`, `dissmfacw_table`).
- Put compiled kernels under `src/`; Python modules should read like an analysis script, not like a build tree.
- User imports should stay on `sequenzo.clustering` unless they are extending an internal submodule.

## Layering

- `clustering`: partitioning workflows plus the compiled backend host
- `utils.core_distance_operations`: shared low-level distance primitives
- `discrepancy_analysis` and other packages: higher-level statistical methods that consume shared primitives
