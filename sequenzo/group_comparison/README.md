# Group Comparison Module

This module contains group-comparison methods for sequence trajectories.

## Scope

Use `sequenzo.group_comparison` for:

- Overall two-group comparison (`LRT`, `BIC`)

Use `sequenzo.discrepancy_analysis` for discrepancy, permutation, and tree-based
distance analysis (`disstree`, `seqtree`) including position-wise/local-window
discrepancy (`seqdiff`-style).

Use `sequenzo.decomposition` for Kitagawa-Oaxaca-Blinder (`KOB`)
decomposition and SA–KOB.

## Why This Exists

Historically, these tools were mixed across modules. The current split keeps
`group_comparison` focused on direct group-comparison workflows only.

## Quick Start

```python
from sequenzo.group_comparison import (
    get_group_differences,
    get_lrt_test,
    get_bic_test,
)
```
