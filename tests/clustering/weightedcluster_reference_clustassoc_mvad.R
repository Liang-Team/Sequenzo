#!/usr/bin/env Rscript
# Reference output for Studer clustassoc vignette (mvad, set.seed(1), LCS, ward.D).
# Usage: Rscript tests/clustering/weightedcluster_reference_clustassoc_mvad.R [outdir]

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else file.path("tests", "clustering", "reference_data")

suppressPackageStartupMessages({
  library(TraMineR)
  library(fastcluster)
  library(WeightedCluster)
})

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

set.seed(1)

data(mvad)

mvad.alphabet <- c("employment", "FE", "HE", "joblessness", "school", "training")
mvad.lab <- c(
  "employment", "further education", "higher education",
  "joblessness", "school", "training"
)
mvad.shortlab <- c("EM", "FE", "HE", "JL", "SC", "TR")

mvad.seq <- seqdef(
  mvad, 17:86,
  alphabet = mvad.alphabet,
  states = mvad.shortlab,
  labels = mvad.lab,
  xtstep = 6
)

diss <- as.matrix(seqdist(mvad.seq, method = "LCS"))
hc <- hclust(as.dist(diss), method = "ward.D")

clustqual <- as.clustrange(hc, diss = diss, ncluster = 10)
cla <- clustassoc(clustqual, diss = diss, covar = mvad$funemp)

write.csv(clustqual$stats, file.path(outdir, "ref_mvad_clustrange_stats.csv"))
write.csv(as.data.frame(clustqual$clustering), file.path(outdir, "ref_mvad_clustrange_clustering.csv"))
write.csv(cla, file.path(outdir, "ref_mvad_clustassoc.csv"))
write.csv(diss, file.path(outdir, "ref_mvad_lcs_diss.csv"))

# Partition labels for k = 5 and k = 6 (for index-plot comparison in tutorial)
for (k in c(5L, 6L)) {
  lab <- cutree(hc, k = k)
  write.csv(
    data.frame(id = mvad$id, cluster = lab),
    file.path(outdir, sprintf("ref_mvad_cluster_k%d.csv", k)),
    row.names = FALSE
  )
}

cat("Wrote reference files to", outdir, "\n")
print(cla)
