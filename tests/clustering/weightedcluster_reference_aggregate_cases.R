#!/usr/bin/env Rscript
# Reference outputs for wcAggregateCases.
# Usage: Rscript tests/clustering/weightedcluster_reference_aggregate_cases.R [outdir]

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(WeightedCluster)
})

x <- data.frame(
  a = c(1, 1, 2, 2, 1),
  b = c("x", "x", "y", "y", "x")
)
weights <- c(1, 2, 1, 3, 4)
result <- wcAggregateCases(x, weights = weights)

out <- data.frame(
  row = seq_len(nrow(x)),
  disaggIndex = result$disaggIndex,
  disaggWeight = result$disaggWeights
)
out <- cbind(out, x)
write.csv(out, file.path(outdir, "ref_aggregate_cases_rows.csv"), row.names = FALSE)

summary <- data.frame(
  aggIndex = result$aggIndex,
  aggWeights = result$aggWeights
)
write.csv(summary, file.path(outdir, "ref_aggregate_cases_summary.csv"), row.names = FALSE)
cat("Wrote aggregate_cases reference files to", outdir, "\n")
