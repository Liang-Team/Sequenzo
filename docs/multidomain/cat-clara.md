# CAT-CLARA

## Concept

**CAT** (cost additive trick) constructs multidomain substitution and indel costs by summing (or averaging) domain-level costs, then runs OM (or HAM/DHD/LCS) on multidomain sequences.

## When to use

- You need compatibility with classic multidomain CAT workflows
- Domain-specific substitution costs are well defined
- Association between domains is moderate (not extreme)

## Example

```python
result = md_clara(
    domains=domains,
    strategy="cat",
    distance_params={
        "method": "OM",
        "sm": ["INDELSLOG", "INDELSLOG", "CONSTANT"],
        "indel": "auto",
        "cweight": [1, 1, 1],
        "link": "sum",
        "norm": "none",
    },
    R=100,
    sample_size=5000,
)
```

## Interpretation

Distances reflect edit costs derived additively from domains. Typologies emphasize domain-cost structure you specify via `sm` and `indel`.

## Common mistakes

- Mismatched `sm` / `indel` list lengths vs number of domains  
- Using CAT when domains are almost deterministic functions of each other — prefer IDCD  
