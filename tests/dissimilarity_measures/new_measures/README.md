# New measures vs TraMineR (OM+INDELS, OM+INDELSLOG, FUTURE, FEATURES, OMtspell)

Tests compare Sequenzo with TraMineR on the **dyadic_children** dataset (same setup as `lcp-lsog` notebook: time columns 15–39, states 1–6, `id_col="dyadID"`). The first 10 rows are used.

## Generating reference matrices (R)

From the **repository root**:

```bash
Rscript tests/dissimilarity_measures/new_measures/traminer_reference.R \
  sequenzo/datasets/dyadic_children.csv \
  10 \
  tests/dissimilarity_measures/new_measures
```

Requires R and TraMineR (`install.packages("TraMineR")`). This writes:

- `ref_om_indels.csv`
- `ref_om_indelslog.csv`
- `ref_om_future.csv`
- `ref_om_features.csv`
- `ref_omtspell.csv`

## Running tests

From the repository root:

```bash
python -m pytest tests/dissimilarity_measures/new_measures/ -v
```

If the ref CSVs are present, tests load them and compare. If not, the fixture tries to run the R script (with a temp copy of the data); if R is unavailable, tests that need refs are skipped.

## Notes

- **OM+INDELS / OM+INDELSLOG**: TraMineR uses state-dependent indel; Sequenzo uses `max(indel)`. Tests only check shape, symmetry, and non-negativity until OM supports vector indel.
- **OM+FUTURE / OM+FEATURES / OMtspell**: Tests use relaxed tolerances (`atol=0.5`, `rtol=0.5`) where implementation details (e.g. alphsize vs R alphabet, padding) can cause small differences.
