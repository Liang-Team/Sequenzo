# Group Comparison Module

This module is the unified entrypoint for methods that compare groups of
sequences or quantify group-level inequality related to sequence trajectories.

## Scope

Use `sequenzo.group_comparison` for:

- Discrepancy-based group association (`pseudo-F`, `pseudo-R2`)
- Permutation testing for distance/group association
- Position-wise group-difference dynamics (`seqdiff`-style)
- Overall two-group comparison (`LRT`, `BIC`)
- Kitagawa-Oaxaca-Blinder decomposition (`KOB`)
- Tree-based subgroup discovery (`disstree`, `seqtree`)

## Why This Exists

Historically, these tools were split between `tree_analysis` and
`compare_differences`. This module provides one conceptual home organized by
research goal (group comparison), not only by algorithm shape.

## Quick Start

```python
from sequenzo.group_comparison import (
    compute_distance_association,
    compare_groups_across_positions,
    compare_groups_overall,
    compute_likelihood_ratio_test,
    compute_bayesian_information_criterion_test,
    kob_decomposition,
    build_sequence_tree,
)
```
