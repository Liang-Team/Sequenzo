# Requirements: Scalable Multidomain Sequence Analysis with CLARA

## 0. Goal

This module implements a scalable framework for multidomain sequence analysis using CLARA.

The framework should support three multidomain dissimilarity strategies:

1. IDCD-CLARA  
   Compute multidomain sequences using the observed expanded alphabet, then compute distances directly at the multidomain level.

2. CAT-CLARA  
   Derive multidomain substitution and indel costs additively from domain-level costs, then compute multidomain distances.

3. DAT-CLARA  
   Compute distances within each domain separately, then combine domain distances additively.

CombT is not included in the core implementation for this paper. It can remain as a future extension.

---

## 1. Existing code base

The current code already provides most building blocks.

### 1.1 Existing multidomain functions

The multidomain module already exposes:

```python
create_idcd_sequence_from_csvs
create_idcd_sequence_from_dfs
compute_cat_distance_matrix
compute_dat_distance_matrix
get_association_between_domains
linked_polyadic_sequence_analysis
```

These are currently imported in the multidomain `__init__.py`. The core functions for this project are IDCD, CAT, and DAT. CombT and linked polyad can stay in the package, but they should not be part of the main paper implementation.

### 1.2 Existing CLARA functions

The CLARA module already has:

```python
clara()
seqclara_range()
```

The current `clara()` already supports:

```python
method = "crisp"
method = "fuzzy"
method = "representativeness"
method = "noise"
```

It also already supports several cluster quality criteria:

```python
distance
db
xb
pbm
ams
```

It also supports stability indicators based on ARI and Jaccard coefficient.

Therefore, the main implementation task is not to rewrite CLARA. The main task is to make IDCD, CAT, and DAT usable inside the CLARA workflow without requiring a full N x N multidomain distance matrix.

---

## 2. Main design principle

The new implementation should use one CLARA engine and three multidomain distance strategies.

The public API should look like this:

```python
md_clara(
    domains,
    strategy="idcd",
    R=100,
    sample_size=5000,
    kvals=range(2, 21),
    method="crisp",
    distance_params=None,
    criteria=("distance", "db", "xb", "pbm", "ams"),
    stability=True,
    random_state=None,
)
```

The user should only need to change:

```python
strategy="idcd"
strategy="cat"
strategy="dat"
```

All three strategies should return comparable output objects.

---

## 3. Target file structure

Add the following files:

```text
sequenzo/
  multidomain/
    idcd.py
    cat.py
    dat.py
    association_between_domains.py
    scalable.py
  big_data/
    clara/
      clara.py
      md_clara.py
      distance_providers.py
      results.py
```

Recommended responsibilities:

```text
multidomain/idcd.py
  Build observed expanded-alphabet multidomain sequences.

multidomain/cat.py
  Build CAT multidomain costs and, where needed, full CAT distance matrices.

multidomain/dat.py
  Build DAT distances from domain distances.

big_data/clara/clara.py
  Existing generic CLARA engine.

big_data/clara/distance_providers.py
  New abstraction for computing only the distances needed by CLARA.

big_data/clara/md_clara.py
  High-level public API for scalable multidomain CLARA.

big_data/clara/results.py
  Result object for storing medoids, labels, quality indices, stability, and settings.
```

Implementation for this project lives under `sequenzo/multidomain/clara/` (reference engine: `sequenzo/big_data/clara/`).

---

## 4–21. (See project issue tracker for full staged plan)

Stages 1–4 are implemented in `sequenzo/multidomain/clara/`:

1. `distance_providers.py` — IDCD, CAT, DAT providers  
2. `clara_engine.py` — provider-aware CLARA (crisp + quality + stability)  
3. `md_clara.py` — public `md_clara()` API  
4. `results.py` — `MDClaraResult` dataclass  

Fuzzy/representativeness, simulation, benchmarks, and docs are planned extensions.
