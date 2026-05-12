#!/usr/bin/env Rscript
# Generate WeightedCluster wcSilhouetteObs reference values for Python tests.
# Usage: Rscript tests/clustering/weightedcluster_reference_silhouette.R [outdir]

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
clustering <- factor(c(1, 1, 1, 2, 2))
weights <- c(1, 2, 1, 1, 3)

asw <- wcSilhouetteObs(diss, clustering, weights = weights, measure = "ASW")
asww <- wcSilhouetteObs(diss, clustering, weights = weights, measure = "ASWw")

out <- data.frame(
  clustering = as.integer(clustering),
  weight = weights,
  ASW = as.numeric(asw),
  ASWw = as.numeric(asww)
)
out <- cbind(out, as.data.frame(diss, check.names = FALSE))

out_path <- file.path(outdir, "ref_observation_silhouette.csv")
write.csv(out, out_path, row.names = FALSE)
cat("Wrote", out_path, "\n")
