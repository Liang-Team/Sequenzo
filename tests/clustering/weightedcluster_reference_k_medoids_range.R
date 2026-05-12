#!/usr/bin/env Rscript
# Reference outputs for wcKMedRange.
# Usage: Rscript tests/clustering/weightedcluster_reference_k_medoids_range.R [outdir]

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(WeightedCluster)
})

set.seed(1)
diss <- as.matrix(
  dist(
    rbind(
      c(0, 0),
      c(0, 1),
      c(1, 0),
      c(3, 3),
      c(4, 4)
    ),
    method = "euclidean"
  )
)
weights <- c(1, 2, 1, 1, 3)
result <- wcKMedRange(diss, kvals = 2:4, weights = weights, method = "PAMonce")

stats <- as.data.frame(result$stats)
clustering <- as.data.frame(result$clustering)
write.csv(stats, file.path(outdir, "ref_k_medoids_range_stats.csv"), row.names = TRUE)
write.csv(clustering, file.path(outdir, "ref_k_medoids_range_clustering.csv"), row.names = FALSE)
cat("Wrote k_medoids_range reference files to", outdir, "\n")
