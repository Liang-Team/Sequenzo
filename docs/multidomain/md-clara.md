# Multidomain CLARA (MD-CLARA)

## Concept

MD-CLARA extends CLARA to multidomain sequence analysis without building a full `N × N` multidomain distance matrix. Distances are computed on demand for each CLARA sample and for assignments to medoids.

Three dissimilarity strategies are supported:

| Strategy | Idea |
|----------|------|
| **IDCD** | Observed expanded-alphabet multidomain sequences; standard OM (or other) distances |
| **CAT** | Additive multidomain substitution/indel costs from domain costs |
| **DAT** | Sum (or mean) of domain-level distance matrices |

## When to use

- Large `N` (thousands to hundreds of thousands of sequences)
- Several domains (2–20+)
- Typology discovery with CLARA sampling

## Quick start

```python
from sequenzo import SequenceData, load_dataset
from sequenzo.multidomain.clara import md_clara

# Build aligned domain SequenceData objects, then:
result = md_clara(
    domains=[seq_a, seq_b, seq_c],
    strategy="idcd",
    distance_params={"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
    R=100,
    sample_size=5000,
    kvals=range(2, 11),
    criteria=("distance",),
    stability=True,
)
```

## Parameters

See `md_clara()` docstring. Key arguments:

- `strategy`: `"idcd"`, `"cat"`, or `"dat"`
- `distance_params`: passed to the distance provider
- `R`, `sample_size`, `kvals`: CLARA design
- `method`: `"crisp"`, `"fuzzy"`, or `"representativeness"`
- `criteria`: quality indices for choosing the best iteration

## Output

`MDClaraResult` provides:

- `result.stats` — quality and stability by `k`
- `result.clustering` — columns `Cluster 2`, …, `Cluster K`
- `result.medoids`, `result.stability`, `result.settings`

## Related pages

- [IDCD-CLARA](idcd-clara.md)
- [CAT-CLARA](cat-clara.md)
- [DAT-CLARA](dat-clara.md)
- [Guidelines](md-clara-guidelines.md)

## Common mistakes

1. Domains with different IDs or time columns → validation error  
2. `sample_size < max(kvals)` → error  
3. IDCD with a huge observed expanded alphabet → slow; check association first  
4. CAT/DAT under strong cross-domain association → biased; prefer IDCD  
