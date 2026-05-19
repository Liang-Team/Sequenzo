# MD-CLARA practical guidelines

## Choosing a strategy

| Situation | Recommendation |
|-----------|----------------|
| Strong state association across domains | **IDCD** |
| Classic CAT paper replication | **CAT** |
| Different metrics per domain | **DAT** |
| Very large expanded alphabet | **IDCD** with monitoring; avoid full-matrix clustering |

## Recommended defaults (exploration)

```python
R=100
sample_size=5000
kvals=range(2, 11)
method="crisp"
criteria=("distance", "db", "ams")
stability=True
```

## Recommended defaults (paper benchmarks)

```python
R=250  # or 500
sample_size=5000  # or 10000
kvals=range(2, 21)
stability=True
```

## Stability

- `stability=True` records ARI/JC across CLARA iterations  
- `ari08` / `jc08`: count of iterations with ARI/JC ≥ 0.8 vs best solution  
- Low stability → increase `R` or `sample_size`

## Benchmarks and simulation

```python
# Add multidomain_clara/ to PYTHONPATH, then:
from simulation import generate_multidomain_sequences, run_md_clara_benchmark

sim = generate_multidomain_sequences(
    n_sequences=1000, n_domains=3, n_timepoints=12,
    alphabet_size=3, n_clusters=4, domain_association="medium",
)
df = run_md_clara_benchmark(n_sequences_grid=(500, 1000), n_domains_grid=(2, 3))
```

## Visualization

```python
from sequenzo.multidomain.clara import (
    plot_md_clara_quality,
    plot_md_clara_stability,
    plot_md_cluster_by_domain,
)

plot_md_clara_quality(result)
plot_md_clara_stability(result)
plot_md_cluster_by_domain(domains, result.clustering["Cluster 4"], k=4)
```

## Warnings the software emits

- Mismatched domain IDs or time axes  
- Expanded alphabet very large (IDCD)  
- `sample_size` small relative to `k`  
