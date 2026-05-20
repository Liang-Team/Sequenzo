# @Author  : Yuqi Liang 梁彧祺
# @File    : weightedcluster_property_reference.R
# @Desc    :
# WeightedCluster seqpropclust reference outputs for property_based_clustering tests.
# Usage: Rscript weightedcluster_property_reference.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example: Rscript weightedcluster_property_reference.R ./dyadic_children.csv 30 .
#
# Writes:
#   ref_propmatrix_state_cols.csv      -- column names from 'state' property block
#   ref_propmatrix_duration.csv        -- 'duration' property values (per-state time)
#   ref_propmatrix_complexity.csv      -- 'Complexity' property values (C, Entropy, Turb, Trans)
#   ref_tree_n_leaves.csv              -- number of leaves in default property tree
#   ref_cut_tree_k2_sizes.csv          -- group sizes for k=2 cut
#   ref_cut_tree_k3_sizes.csv          -- group sizes for k=3 cut
#   ref_prune_k2_sizes.csv             -- group sizes after dtprune to k=2
#   ref_quality_pbc.csv                -- PBC quality indicator (k=3..8)
#   ref_quality_r2.csv                 -- R2 quality indicator (k=3..8)

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}
if (!requireNamespace("WeightedCluster", quietly = TRUE)) {
  stop("WeightedCluster required: install.packages('WeightedCluster')")
}

args    <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript weightedcluster_property_reference.R <csv_path> [nrows=30] [outdir=.]")
csv_path <- args[1]
nrows    <- if (length(args) >= 2) as.integer(args[2]) else 30L
outdir   <- if (length(args) >= 3) args[3] else "."

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
dat       <- read.csv(csv_path, nrows = nrows, check.names = FALSE)
time_cols <- names(dat)[sapply(names(dat), function(x) suppressWarnings(!is.na(as.numeric(x))))]
time_cols <- time_cols[order(as.numeric(time_cols))]
id_col    <- "dyadID"
states    <- c(1, 2, 3, 4, 5, 6)

suppressMessages({
  seqdata <- TraMineR::seqdef(dat[, time_cols, drop = FALSE],
                               alphabet = states,
                               id       = dat[[id_col]])
  diss <- TraMineR::seqdist(seqdata, method = "OM", sm = "TRATE",
                             indel = "auto", norm = "maxlength")
})
cat("Loaded", nrows, "sequences, dist range [", min(diss), ",", max(diss), "]\n")

# ---------------------------------------------------------------------------
# Part 1: Property extraction
# ---------------------------------------------------------------------------

# 'state' block: seqpropclust(prop.only=TRUE) — seqistat is not exported in TraMineR 2.2+
cat("Extracting 'state' properties...\n")
state_props <- suppressMessages(
  WeightedCluster::seqpropclust(seqdata, diss = diss, properties = "state",
                                  prop.only = TRUE, R = 1)
)
write.csv(state_props,
          file.path(outdir, "ref_propmatrix_state_cols.csv"), row.names = TRUE)

# 'duration' block: seqistatd (total time in each state)
cat("Extracting 'duration' properties...\n")
dur_props <- TraMineR::seqistatd(seqdata)
write.csv(dur_props,
          file.path(outdir, "ref_propmatrix_duration.csv"), row.names = TRUE)

# 'Complexity' block: C, Entropy, Turbulence, number of transitions
cat("Extracting 'Complexity' properties...\n")
complexity_c <- get("seqici", envir = asNamespace("WeightedCluster"))(seqdata)
complexity_props <- data.frame(
  C          = complexity_c,
  Entropy    = TraMineR::seqient(seqdata),
  Turbulence = TraMineR::seqST(seqdata),
  Trans.     = TraMineR::seqtransn(seqdata)
)
write.csv(complexity_props,
          file.path(outdir, "ref_propmatrix_complexity.csv"), row.names = TRUE)

# ---------------------------------------------------------------------------
# Part 2: Full seqpropclust tree (state + duration properties)
# ---------------------------------------------------------------------------
cat("Running seqpropclust (state + duration)...\n")
set.seed(42)
pclust <- suppressMessages(
  WeightedCluster::seqpropclust(
    seqdata, diss = diss,
    properties = c("state", "duration"),
    R = 1,
    weight.permutation = "diss",
    min.size = 0.01,
    max.depth = 5,
    pval = 1.0
  )
)
# Match Sequenzo: property_based_clustering always applies cluster_split_schedule.
cluster_split_schedule <- get("clusterSplitSchedule", envir = asNamespace("WeightedCluster"))
pclust <- cluster_split_schedule(pclust)
cat("Tree built. Info:\n")
print(pclust$info)

# Number of leaf groups
fitted_all  <- pclust$fitted[["(fitted)"]]
n_leaves    <- length(unique(fitted_all))
write.csv(data.frame(n_leaves = n_leaves),
          file.path(outdir, "ref_tree_n_leaves.csv"), row.names = FALSE)
cat("n_leaves:", n_leaves, "\n")

# ---------------------------------------------------------------------------
# Part 3: dtcut (cut_tree) partition sizes
# ---------------------------------------------------------------------------
cat("Computing cut_tree partitions...\n")

# k = 2
if (n_leaves >= 2) {
  cut2       <- WeightedCluster::dtcut(pclust, k = 2)
  sizes_k2   <- as.integer(table(cut2))
  write.csv(data.frame(size = sort(sizes_k2)),
            file.path(outdir, "ref_cut_tree_k2_sizes.csv"), row.names = FALSE)
  cat("k=2 sizes:", sort(sizes_k2), "\n")
}

# k = 3
if (n_leaves >= 3) {
  cut3       <- WeightedCluster::dtcut(pclust, k = 3)
  sizes_k3   <- as.integer(table(cut3))
  write.csv(data.frame(size = sort(sizes_k3)),
            file.path(outdir, "ref_cut_tree_k3_sizes.csv"), row.names = FALSE)
  cat("k=3 sizes:", sort(sizes_k3), "\n")
}

# ---------------------------------------------------------------------------
# Part 4: dtprune (prune_property_tree)
# ---------------------------------------------------------------------------
cat("Computing dtprune k=2...\n")
if (n_leaves >= 2) {
  pruned2     <- get("dtprune", envir = asNamespace("WeightedCluster"))(pclust, k = 2, diss = diss)
  fitted_p2   <- pruned2$fitted[["(fitted)"]]
  sizes_p2    <- as.integer(table(fitted_p2))
  write.csv(data.frame(size = sort(sizes_p2)),
            file.path(outdir, "ref_prune_k2_sizes.csv"), row.names = FALSE)
  cat("Pruned k=2 sizes:", sort(sizes_p2), "\n")
}

# ---------------------------------------------------------------------------
# Part 5: as.clustrange quality indicators
# ---------------------------------------------------------------------------
cat("Computing cluster quality (as.clustrange)...\n")
n_eval <- min(n_leaves, 8L)

if (n_eval >= 3) {
  quality <- suppressMessages(
    WeightedCluster::as.clustrange(pclust, diss = diss, ncluster = n_eval)
  )
  stats   <- quality$stats  # data.frame with quality indicators per k

  pbc_vals <- stats[["PBC"]]
  write.csv(data.frame(PBC = pbc_vals),
            file.path(outdir, "ref_quality_pbc.csv"), row.names = TRUE)

  r2_vals <- stats[["R2"]]
  write.csv(data.frame(R2 = r2_vals),
            file.path(outdir, "ref_quality_r2.csv"), row.names = TRUE)

  cat("Quality stats (PBC, R2):\n")
  print(stats[, c("PBC", "R2")])
} else {
  cat("Not enough leaves for quality evaluation (need >= 3).\n")
}

cat("\nAll reference files written to:", outdir, "\n")
