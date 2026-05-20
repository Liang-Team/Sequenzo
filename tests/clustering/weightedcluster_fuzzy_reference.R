# @Author  : Yuqi Liang 梁彧祺
# @File    : weightedcluster_fuzzy_reference.R
# @Desc    :
# WeightedCluster / cluster::fanny reference outputs for fuzzy_clustering tests.
# Usage: Rscript weightedcluster_fuzzy_reference.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example: Rscript weightedcluster_fuzzy_reference.R ./dyadic_children.csv 20 .
#
# Writes:
#   ref_fanny_k3_exp15_memb.csv          -- FANNY k=3, memb.exp=1.5 membership matrix
#   ref_fanny_k3_exp15_obj.csv           -- FANNY k=3, memb.exp=1.5 objective value
#   ref_fanny_k3_exp15_crispness.csv     -- normalized crispness per sequence
#   ref_fanny_k3_exp15_memb_summary.csv  -- summary() of membership (6 stats x k)
#   ref_fanny_k3_exp15_most_typical.csv  -- index of most typical member per cluster (1-indexed)
#   ref_fanny_k3_exp20_memb.csv          -- FANNY k=3, memb.exp=2.0 membership
#   ref_fanny_k2_exp15_memb.csv          -- FANNY k=2, memb.exp=1.5 membership
#   ref_fanny_k4_exp15_memb.csv          -- FANNY k=4, memb.exp=1.5 membership
#   ref_wfcmdd_fcmdd_k3_memb.csv         -- wfcmdd FCMdd k=3, m=2.0 membership
#   ref_wfcmdd_fcmdd_k3_functional.csv   -- wfcmdd FCMdd k=3, m=2.0 functional value
#   ref_wfcmdd_ncdd_k3_memb.csv          -- wfcmdd NCdd k=3 membership (incl. noise col)
#   ref_wfcmdd_hncdd_k3_memb.csv         -- wfcmdd HNCdd k=3 membership
#   ref_wfcmdd_pcmdd_k3_memb.csv         -- wfcmdd PCMdd k=3 membership
#   ref_wfcmdd_fcmdd_k3_m15_memb.csv     -- wfcmdd FCMdd k=3, m=1.5 membership
#   ref_wfcmdd_fcmdd_k3_m30_memb.csv     -- wfcmdd FCMdd k=3, m=3.0 membership

if (!requireNamespace("cluster", quietly = TRUE)) {
  stop("Package 'cluster' is required: install.packages('cluster')")
}
if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("Package 'TraMineR' is required: install.packages('TraMineR')")
}
if (!requireNamespace("WeightedCluster", quietly = TRUE)) {
  stop("Package 'WeightedCluster' is required: install.packages('WeightedCluster')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript weightedcluster_fuzzy_reference.R <csv_path> [nrows=20] [outdir=.]")
csv_path <- args[1]
nrows    <- if (length(args) >= 2) as.integer(args[2]) else 20L
outdir   <- if (length(args) >= 3) args[3] else "."

# ---------------------------------------------------------------------------
# Load data and build distance matrix (same as Python side)
# ---------------------------------------------------------------------------
dat       <- read.csv(csv_path, nrows = nrows, check.names = FALSE)
time_cols <- names(dat)[sapply(names(dat), function(x) suppressWarnings(!is.na(as.numeric(x))))]
time_cols <- time_cols[order(as.numeric(time_cols))]
id_col    <- "dyadID"
states    <- c(1, 2, 3, 4, 5, 6)

suppressMessages({
  seqdata <- TraMineR::seqdef(dat[, time_cols, drop = FALSE],
                               alphabet = states,
                               id = dat[[id_col]])
  diss <- TraMineR::seqdist(seqdata, method = "OM", sm = "TRATE",
                             indel = "auto", norm = "maxlength")
})

cat("Data loaded:", nrows, "sequences,", ncol(seqdata), "time points\n")
cat("Distance matrix range: [", min(diss), ",", max(diss), "]\n")

# ---------------------------------------------------------------------------
# Helper: WeightedCluster crispness (mirrors wc.crispness)
# ---------------------------------------------------------------------------
wc_crispness <- function(memb, norm = TRUE) {
  k <- ncol(memb)
  values <- rowSums(memb^2)
  if (!norm || k <= 1) return(values)
  (values - 1 / k) / (1 - 1 / k)
}

# Helper: most typical member per cluster (highest membership per column)
most_typical <- function(memb) {
  apply(memb, 2, which.max)  # returns 1-based row index
}

# Helper: membership summary (mirrors R summary() on data.frame)
memb_summary <- function(memb) {
  df <- as.data.frame(memb)
  do.call(cbind, lapply(df, function(col) {
    c("Min."    = min(col),
      "1st Qu." = quantile(col, 0.25, names = FALSE),
      "Median"  = median(col),
      "Mean"    = mean(col),
      "3rd Qu." = quantile(col, 0.75, names = FALSE),
      "Max."    = max(col))
  }))
}

# ---------------------------------------------------------------------------
# Seed for reproducibility of fanny (uses random initializations internally)
# ---------------------------------------------------------------------------
set.seed(42)

# =============================================================================
# FANNY (cluster::fanny)
# =============================================================================

cat("\n--- FANNY k=3, memb.exp=1.5 ---\n")
fanny_k3_e15 <- cluster::fanny(diss, k = 3, diss = TRUE, memb.exp = 1.5)
write.csv(fanny_k3_e15$membership,
          file.path(outdir, "ref_fanny_k3_exp15_memb.csv"), row.names = TRUE)
write.csv(data.frame(objective = fanny_k3_e15$objective[["objective"]]),
          file.path(outdir, "ref_fanny_k3_exp15_obj.csv"), row.names = FALSE)

crisp_k3_e15 <- wc_crispness(fanny_k3_e15$membership, norm = TRUE)
write.csv(data.frame(crispness = crisp_k3_e15),
          file.path(outdir, "ref_fanny_k3_exp15_crispness.csv"), row.names = TRUE)

summ_k3_e15 <- memb_summary(fanny_k3_e15$membership)
write.csv(summ_k3_e15,
          file.path(outdir, "ref_fanny_k3_exp15_memb_summary.csv"), row.names = TRUE)

typical_k3_e15 <- most_typical(fanny_k3_e15$membership)
write.csv(data.frame(index = typical_k3_e15),
          file.path(outdir, "ref_fanny_k3_exp15_most_typical.csv"), row.names = FALSE)

cat("--- FANNY k=3, memb.exp=2.0 ---\n")
set.seed(42)
fanny_k3_e20 <- cluster::fanny(diss, k = 3, diss = TRUE, memb.exp = 2.0)
write.csv(fanny_k3_e20$membership,
          file.path(outdir, "ref_fanny_k3_exp20_memb.csv"), row.names = TRUE)

cat("--- FANNY k=2, memb.exp=1.5 ---\n")
set.seed(42)
fanny_k2_e15 <- cluster::fanny(diss, k = 2, diss = TRUE, memb.exp = 1.5)
write.csv(fanny_k2_e15$membership,
          file.path(outdir, "ref_fanny_k2_exp15_memb.csv"), row.names = TRUE)

cat("--- FANNY k=4, memb.exp=1.5 ---\n")
set.seed(42)
fanny_k4_e15 <- cluster::fanny(diss, k = 4, diss = TRUE, memb.exp = 1.5)
write.csv(fanny_k4_e15$membership,
          file.path(outdir, "ref_fanny_k4_exp15_memb.csv"), row.names = TRUE)

# =============================================================================
# wfcmdd (internal in WeightedCluster 1.8.x — not exported, use :::)
# =============================================================================
wfcmdd <- WeightedCluster:::wfcmdd

# Seeds must match the Python side: seeds = np.array([0, 5, 10]) → R 1-based: c(1, 6, 11)
seeds_r  <- c(1, 6, 11)   # 1-based row indices into diss
dnoise_r <- median(diss[diss > 0])
n_seq    <- attr(diss, "Size")
if (is.null(n_seq)) n_seq <- nrow(as.matrix(diss))
weights_r <- rep(1, n_seq)  # WeightedCluster 1.8.x wfcmdd requires explicit weights

cat("\n--- wfcmdd FCMdd k=3, m=2.0 ---\n")
set.seed(42)
wfc_fcmdd_k3 <- wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "FCMdd", m = 2.0)
write.csv(wfc_fcmdd_k3$memb,
          file.path(outdir, "ref_wfcmdd_fcmdd_k3_memb.csv"), row.names = TRUE)
write.csv(data.frame(functional = wfc_fcmdd_k3$functional),
          file.path(outdir, "ref_wfcmdd_fcmdd_k3_functional.csv"), row.names = FALSE)

cat("--- wfcmdd NCdd k=3 ---\n")
set.seed(42)
wfc_ncdd_k3 <- wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "NCdd", m = 2.0, dnoise = dnoise_r)
write.csv(wfc_ncdd_k3$memb,
          file.path(outdir, "ref_wfcmdd_ncdd_k3_memb.csv"), row.names = TRUE)

cat("--- wfcmdd HNCdd k=3 ---\n")
set.seed(42)
wfc_hncdd_k3 <- wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "HNCdd", dnoise = dnoise_r)
write.csv(wfc_hncdd_k3$memb,
          file.path(outdir, "ref_wfcmdd_hncdd_k3_memb.csv"), row.names = TRUE)

# PCMdd on dyadic_children often fails in WeightedCluster 1.8.x (subscript out of bounds).
# pytest uses tests/clustering/ref_wfcmdd_pcmdd_k3_memb.csv from mvad + Sequenzo wfcmdd (HAM).
cat("--- wfcmdd PCMdd k=3 (optional; may fail on this data) ---\n")
set.seed(42)
eta_r <- rep(0.5, 3)
wfc_pcmdd_k3 <- tryCatch(
  wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "PCMdd", m = 2.0, eta = eta_r),
  error = function(e) {
    cat("WARNING: PCMdd wfcmdd failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(wfc_pcmdd_k3)) {
  write.csv(wfc_pcmdd_k3$memb,
            file.path(outdir, "ref_wfcmdd_pcmdd_k3_memb.csv"), row.names = TRUE)
}

cat("--- wfcmdd FCMdd k=3, m=1.5 ---\n")
set.seed(42)
wfc_fcmdd_k3_m15 <- tryCatch(
  wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "FCMdd", m = 1.5),
  error = function(e) {
    cat("WARNING: FCMdd m=1.5 wfcmdd failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(wfc_fcmdd_k3_m15)) {
  write.csv(wfc_fcmdd_k3_m15$memb,
            file.path(outdir, "ref_wfcmdd_fcmdd_k3_m15_memb.csv"), row.names = TRUE)
}

cat("--- wfcmdd FCMdd k=3, m=3.0 ---\n")
set.seed(42)
wfc_fcmdd_k3_m30 <- tryCatch(
  wfcmdd(diss, memb = seeds_r, weights = weights_r, method = "FCMdd", m = 3.0),
  error = function(e) {
    cat("WARNING: FCMdd m=3.0 wfcmdd failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(wfc_fcmdd_k3_m30)) {
  write.csv(wfc_fcmdd_k3_m30$memb,
            file.path(outdir, "ref_wfcmdd_fcmdd_k3_m30_memb.csv"), row.names = TRUE)
}

cat("\nAll reference matrices written to:", outdir, "\n")
