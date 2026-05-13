# Decomposition

Decompose differences in an outcome between two groups into explained and
unexplained components using Kitagawa–Oaxaca–Blinder (KOB) and related methods.

Although these tools are widely used for inequality analysis (wage gaps, pension
gaps, health gaps), the methods apply to any binary group comparison of a
continuous outcome.

## Module layout

| Layer | Module | Entry points |
|-------|--------|--------------|
| Engine | `oaxaca.py` | `get_oaxaca_blinder_decomposition()` |
| Generic KOB | `kob.py` | `get_kob_decomposition()`, `get_kob_decomposition_bootstrap()` |
| SA–KOB | `sa_kob.py` | `get_sa_kob_decomposition()`, `get_sa_kob_decomposition_bootstrap()` |
| Results | `results.py` | `KOBDecompositionResult`, `SAKOBDecompositionResult`, … |

## Quick start (generic KOB)

```python
from sequenzo.decomposition import get_kob_decomposition

result = get_kob_decomposition(
    y=outcome,
    group=group,
    X=covariates,
    group0_value="men",
    group1_value="women",
)

result.total_gap       # mean(group0) - mean(group1)
result.explained       # composition effect
result.unexplained_returns
result.by_column       # dummy-column contributions
result.by_category     # full category-level contributions when normalized
result.by_term         # term-level aggregates
```

Positive `total_gap` means `group0` has a higher mean outcome than `group1`.

`by_category` is populated only for categorical terms normalized for detailed
decomposition (`normalize_categorical=True`). With continuous covariates only,
it stays empty; use `by_column` or `by_term` instead.

When `normalize_categorical=True` and `categorical_terms` is omitted, Sequenzo
auto-detects terms with more than one column as categorical and emits a
warning. Pass `categorical_terms` explicitly for non-dummy multi-column terms.

Scalar `explained` and `unexplained_returns` keep the raw twofold totals so the
decomposition identity holds. Detailed tables may sum differently when
mixed-reference normalization is used.

Generic bootstrap uses stratified resampling by default so each draw keeps the
original group sizes. All confidence intervals respect `confidence_level`.

## Quick start (SA–KOB with sequence clusters)

For life-course typologies from sequence analysis (Rowold, Struffolino, and
Fasang, 2025), use the SA–KOB wrapper. It builds cluster dummies with internal
category ids `0 .. k-1`, implements a practical majority-rule version of
Rowold, Struffolino, and Fasang’s cluster-specific reference-coefficient
strategy (option III), and returns all `k` clusters in `by_cluster`, including
the omitted baseline cluster.

```python
from sequenzo.decomposition import get_sa_kob_decomposition

result = get_sa_kob_decomposition(
    y=pension_income,
    group=sex,
    cluster_labels=life_course_cluster,
    k=8,
    reference_category_index=0,
    cluster_coefficient_reference="majority",
    fallback_reference="group0",
    group0_value="men",
    group1_value="women",
)

result.by_cluster              # all k clusters, including baseline
result.cluster_composition
result.cluster_owners          # owners for all k clusters
result.common_support_table
result.explained_detailed      # sum of by_cluster explained (Yun-normalized)
result.returns_detailed        # sum of by_cluster returns (Yun-normalized)
result.explained_difference    # explained_detailed - explained
result.returns_difference      # returns_detailed - unexplained_returns
```

The scalar `explained` and `unexplained_returns` follow the raw twofold
decomposition. The cluster-level table reports Yun-normalized,
reference-invariant category attribution, so its summed detailed contributions
(`explained_detailed`, `returns_detailed`) may differ slightly from the raw
scalar components in mixed-reference settings.

Parameter notes:

- `cluster_coefficient_reference` must be `"majority"`, `"group0"`, `"group1"`,
  or `"pooled"` (Rowold et al. options III, I, and II).
- `fallback_reference` must be `"group0"`, `"group1"`, or `"pooled"`. It applies
  to non-cluster controls and coefficients coded with owner `-1`.
- `cluster_coefficient_reference="pooled"` automatically sets
  `fallback_reference="pooled"`.
- `neutral_cluster_owner` is `0` or `1` by default for neutral clusters (option III);
  set to `None` to route neutral clusters through `fallback_reference`.
- `cluster_owner_overrides` may be keyed by original cluster labels or internal
  category ids; original labels take precedence when both could apply.

## Bootstrap uncertainty

```python
from sequenzo.decomposition import get_sa_kob_decomposition_bootstrap

boot = get_sa_kob_decomposition_bootstrap(
    y=y,
    group=group,
    cluster_labels=clusters,
    k=8,
    n_boot=500,
    random_state=42,
    stratified=True,
)
boot.by_cluster_confidence_intervals
```

`by_cluster_standard_errors` and `by_cluster_confidence_intervals` return the
`by_cluster` table augmented with SE or CI columns.

Bootstrap draws fix the full category universe (from the point estimate) so
rare clusters still produce valid dummy columns even when absent from a resample.
You usually do not need to pass `categories=` explicitly.

## Related modules

- `sequenzo.group_comparison` — overall two-group tests (LRT, BIC)
- `sequenzo.clustering` — sequence typologies and cluster-quality diagnostics

## Reference

Rowold, C., Struffolino, E., & Fasang, A. E. (2025). Life-course-sensitive analysis of group inequalities: Combining sequence analysis with the Kitagawa–Oaxaca–Blinder decomposition. Sociological Methods & Research, 54(2), 646-705.
