# DAT-CLARA

## Concept

**DAT** (distance additive trick) computes a distance matrix in each domain, then combines them with weights (`link="sum"` or `"mean"`).

## When to use

- Different domains need different dissimilarity measures (e.g. OM + HAM + DHD)
- You want transparent domain weights in the typology
- Domains are weakly associated

## Example

```python
result = md_clara(
    domains=domains,
    strategy="dat",
    distance_params={
        "method_params": [
            {"method": "OM", "sm": "INDELSLOG", "indel": "auto"},
            {"method": "OM", "sm": "INDELSLOG", "indel": "auto"},
            {"method": "HAM"},
        ],
        "domain_weights": [1, 1, 1],
        "link": "sum",
    },
    R=100,
    sample_size=5000,
)
```

## Interpretation

Clusters merge individuals who are close in **weighted sum of domain distances**. Dominant domains (large weights or large scales) drive the solution.

## Common mistakes

- `method_params` length ≠ number of domains  
- Ignoring scale differences between domains — normalize or adjust `domain_weights`  
