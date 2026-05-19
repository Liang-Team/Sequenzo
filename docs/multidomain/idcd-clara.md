# IDCD-CLARA

## Concept

**IDCD** (independence from domain costs and distances) builds multidomain sequences from **observed** combined states only, then computes dissimilarities directly on that expanded alphabet.

## When to use

- Domains are associated (states co-occur in predictable ways)
- You want a multidomain representation that does not rely on additive cost/distance tricks
- Paper-recommended default for substantive multidomain typologies

## Example

```python
result = md_clara(
    domains=[seq_left, seq_child, seq_married],
    strategy="idcd",
    distance_params={
        "method": "OM",
        "sm": "INDELSLOG",
        "indel": "auto",
        "norm": "none",
    },
    R=100,
    sample_size=5000,
    kvals=range(2, 11),
)
```

## Interpretation

Clusters are groups of individuals with similar **joint** state trajectories. The expanded alphabet size reflects how many combined states were actually observed.

## Common mistakes

- Expecting all Cartesian products of domain states to appear — IDCD only uses observed combinations  
- Very large expanded alphabet → increase `sample_size` and `R`, or simplify domains  
