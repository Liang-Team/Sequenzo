# @Author  : Yuqi Liang 梁彧祺
# @File    : traminer_twed_reference.R
# @Time    : 2026/02/07 22:40
# @Desc    :
# TraMineR reference distances for attribute-based measures:
#   LCS, NMS (with prox), NMSMST, NMSSTSSoft (NMS with proximity; same as NMS with prox here), SVRspell.
# Uses only the first 10 time columns to avoid C++ overflow in NMS/SVRspell (subsequence sums).
# Usage: Rscript traminer_reference_attribute.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example (from repo root): Rscript tests/dissimilarity_measures/new_measures/traminer_reference_attribute.R sequenzo/datasets/dyadic_children.csv 10 tests/dissimilarity_measures/new_measures
# Example (from new_measures/): Rscript traminer_reference_attribute.R ../../../sequenzo/datasets/dyadic_children.csv 10 .
# Writes: ref_lcs.csv, ref_nms.csv, ref_nmsmst.csv, ref_nmsstssoft.csv, ref_svrspell.csv

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript traminer_reference_attribute.R <csv_path> [nrows=10] [outdir=.]")
csv_path <- args[1]
nrows <- if (length(args) >= 2) as.integer(args[2]) else 10L
outdir <- if (length(args) >= 3) args[3] else "."

dat <- read.csv(csv_path, nrows = nrows, check.names = FALSE)
time_cols <- names(dat)[sapply(names(dat), function(x) suppressWarnings(!is.na(as.numeric(x))))]
time_cols <- time_cols[order(as.numeric(time_cols))]
# Use only first 10 time columns to avoid "Number of subsequences is getting too big" in NMS/SVRspell C++ code
max_time_cols <- 10L
time_cols <- head(time_cols, max_time_cols)
id_col <- "dyadID"
states <- c(1, 2, 3, 4, 5, 6)
nstates <- length(states)

seqdata <- TraMineR::seqdef(dat[, time_cols, drop = FALSE], alphabet = states, id = dat[[id_col]])

# TraMineR allows only norm = "none" for NMS, NMSMST, SVRspell (no other norm option).
# LCS: we use norm = "none" to match raw distance.

# ----- LCS (norm=none) -----
suppressMessages({
  D_lcs <- TraMineR::seqdist(seqdata, method = "LCS", norm = "none")
})
write.csv(as.matrix(D_lcs), file.path(outdir, "ref_lcs.csv"), row.names = TRUE)

# ----- NMS with prox = identity (soft matching). TraMineR requires prox for NMS. -----
prox_identity <- diag(nstates)
suppressMessages({
  D_nms <- TraMineR::seqdist(seqdata, method = "NMS", prox = prox_identity, norm = "none")
})
write.csv(as.matrix(D_nms), file.path(outdir, "ref_nms.csv"), row.names = TRUE)
write.csv(as.matrix(D_nms), file.path(outdir, "ref_nmsstssoft.csv"), row.names = TRUE)

# ----- NMSMST (norm=none) -----
suppressMessages({
  D_nmsmst <- TraMineR::seqdist(seqdata, method = "NMSMST", norm = "none")
})
write.csv(as.matrix(D_nmsmst), file.path(outdir, "ref_nmsmst.csv"), row.names = TRUE)

# ----- SVRspell (default prox = identity in TraMineR, norm=none) -----
suppressMessages({
  D_svrspell <- TraMineR::seqdist(seqdata, method = "SVRspell", norm = "none")
})
write.csv(as.matrix(D_svrspell), file.path(outdir, "ref_svrspell.csv"), row.names = TRUE)

# ----- Part 2b: Attribute parameter configs (for TraMineR comparison) -----
# LCS norm = "gmean"
suppressMessages({
  D_lcs_gmean <- TraMineR::seqdist(seqdata, method = "LCS", norm = "gmean")
})
write.csv(as.matrix(D_lcs_gmean), file.path(outdir, "ref_lcs_gmean.csv"), row.names = TRUE)

# NMS with prox = 0.5*I + (1-0.5)/nstates (off-diagonal constant)
prox_offdiag <- diag(nstates) * 0.5 + (1.0 - 0.5) / nstates
suppressMessages({
  D_nms_prox <- TraMineR::seqdist(seqdata, method = "NMS", prox = prox_offdiag, norm = "none")
})
write.csv(as.matrix(D_nms_prox), file.path(outdir, "ref_nms_prox_offdiag.csv"), row.names = TRUE)

# NMSMST with tpow = 0.5 and 2.0
suppressMessages({
  D_nmsmst_t05 <- TraMineR::seqdist(seqdata, method = "NMSMST", norm = "none", tpow = 0.5)
})
write.csv(as.matrix(D_nmsmst_t05), file.path(outdir, "ref_nmsmst_tpow05.csv"), row.names = TRUE)
suppressMessages({
  D_nmsmst_t2 <- TraMineR::seqdist(seqdata, method = "NMSMST", norm = "none", tpow = 2.0)
})
write.csv(as.matrix(D_nmsmst_t2), file.path(outdir, "ref_nmsmst_tpow2.csv"), row.names = TRUE)

# SVRspell with tpow = 0.5 and 2.0
suppressMessages({
  D_svr_t05 <- TraMineR::seqdist(seqdata, method = "SVRspell", norm = "none", tpow = 0.5)
})
write.csv(as.matrix(D_svr_t05), file.path(outdir, "ref_svrspell_tpow05.csv"), row.names = TRUE)
suppressMessages({
  D_svr_t2 <- TraMineR::seqdist(seqdata, method = "SVRspell", norm = "none", tpow = 2.0)
})
write.csv(as.matrix(D_svr_t2), file.path(outdir, "ref_svrspell_tpow2.csv"), row.names = TRUE)

# NMSMST with custom kweights: first position 2, rest 0.5 (length = ncol(seqdata) = max_time_cols)
kweights_custom <- rep(0.5, ncol(seqdata))
kweights_custom[1] <- 2.0
suppressMessages({
  D_nmsmst_kw <- TraMineR::seqdist(seqdata, method = "NMSMST", norm = "none", kweights = kweights_custom)
})
write.csv(as.matrix(D_nmsmst_kw), file.path(outdir, "ref_nmsmst_kweights.csv"), row.names = TRUE)

cat("Attribute-based reference matrices written to", outdir, "\n")
