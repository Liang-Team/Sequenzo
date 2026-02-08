# Dissimilarity measures vs TraMineR

Unified tests in **test_dissimilarity_measures_traminer.py** compare Sequenzo with TraMineR for:

1. **OM variants** (Part 1): OM+INDELS, OM+INDELSLOG, FUTURE, FEATURES, OMtspell.
2. **OM parameter configs** (Part 1b): norm=none, scalar indel, norm=gmean, expcost=0.3/0.7.
3. **Attribute-based** (Part 2): LCS, NMS, NMSMST, NMSSTSSoft, SVRspell.
4. **Attribute parameter configs** (Part 2b): LCS gmean, NMS custom prox, NMSMST/SVRspell tpow, NMSMST kweights.
5. **TWED** (Part 3): base config (hardcoded ref); **Part 3b**: nu/h/norm/indel variants vs R ref CSVs.

Data: dyadic_children (first 10 rows; attribute tests use first 10 time columns). TWED: synthetic 4×5 sequences.

## Generating reference matrices (R)

From the **repository root**. Requires R and TraMineR (`install.packages("TraMineR")`).

**OM (Parts 1 + 1b):**

```bash
Rscript tests/dissimilarity_measures/new_measures/traminer_reference.R \
  sequenzo/datasets/dyadic_children.csv 10 tests/dissimilarity_measures/new_measures
```

**Attribute (Parts 2 + 2b):**

```bash
Rscript tests/dissimilarity_measures/new_measures/traminer_reference_attribute.R \
  sequenzo/datasets/dyadic_children.csv 10 tests/dissimilarity_measures/new_measures
```

**TWED (Part 3b):**

```bash
Rscript tests/dissimilarity_measures/new_measures/traminer_twed_reference.R \
  tests/dissimilarity_measures/new_measures
```

If ref CSVs are missing, fixtures try to run the R scripts; if R is unavailable, the corresponding tests are skipped.

## Running tests

```bash
pytest tests/dissimilarity_measures/new_measures/test_dissimilarity_measures_traminer.py -v
```

## Notes

- **OM variants (INDELS, INDELSLOG, FUTURE, FEATURES, OMtspell)**: Implementations are aligned with TraMineR (seqcost, Gower/daisy for FEATURES, OMPerdistanceII for OMtspell). Tests use tight tolerance (`atol=1e-6`, `rtol=1e-5`) so results are strictly consistent.
- **Attribute-based**: First 10 time columns only to avoid C++ overflow in NMS/SVRspell.
