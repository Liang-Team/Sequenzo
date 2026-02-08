# TraMineR reference distances for: OM+INDELS, OM+INDELSLOG, OM+FUTURE, OM+FEATURES, OMtspell.
# Usage: Rscript traminer_reference.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example: Rscript traminer_reference.R ../../../sequenzo/datasets/dyadic_children.csv 10 .
# Writes: ref_om_indels.csv, ref_om_indelslog.csv, ref_om_future.csv, ref_om_features.csv, ref_omtspell.csv

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript traminer_reference.R <csv_path> [nrows=10] [outdir=.]")
csv_path <- args[1]
nrows <- if (length(args) >= 2) as.integer(args[2]) else 10L
outdir <- if (length(args) >= 3) args[3] else "."

dat <- read.csv(csv_path, nrows = nrows, check.names = FALSE)
# Time columns: numeric year names 15..39 (as in lcp-lsog notebook)
time_cols <- names(dat)[sapply(names(dat), function(x) suppressWarnings(!is.na(as.numeric(x))))]
time_cols <- time_cols[order(as.numeric(time_cols))]
id_col <- "dyadID"
states <- c(1, 2, 3, 4, 5, 6)

# Use only sequence columns so dyadID/sex are not interpreted as states
seqdata <- TraMineR::seqdef(dat[, time_cols, drop = FALSE], alphabet = states, id = dat[[id_col]])

# ----- OM + INDELS (norm=maxlength to match typical auto) -----
suppressMessages({
  D_indels <- TraMineR::seqdist(seqdata, method = "OM", sm = "INDELS", indel = "auto", norm = "maxlength")
})
write.csv(as.matrix(D_indels), file.path(outdir, "ref_om_indels.csv"), row.names = TRUE)

# ----- OM + INDELSLOG -----
suppressMessages({
  D_indelslog <- TraMineR::seqdist(seqdata, method = "OM", sm = "INDELSLOG", indel = "auto", norm = "maxlength")
})
write.csv(as.matrix(D_indelslog), file.path(outdir, "ref_om_indelslog.csv"), row.names = TRUE)

# ----- OM + FUTURE (sm from seqcost FUTURE) -----
suppressMessages({
  sc_future <- TraMineR::seqcost(seqdata, method = "FUTURE", with.missing = FALSE)
  D_future <- TraMineR::seqdist(seqdata, method = "OM", sm = sc_future$sm, indel = sc_future$indel, norm = "maxlength")
})
write.csv(as.matrix(D_future), file.path(outdir, "ref_om_future.csv"), row.names = TRUE)

# ----- OM + FEATURES (state.features: one row per state, 6 states) -----
state_features <- data.frame(f1 = seq_along(states))
suppressMessages({
  sc_feat <- TraMineR::seqcost(seqdata, method = "FEATURES", state.features = state_features, with.missing = FALSE)
  D_features <- TraMineR::seqdist(seqdata, method = "OM", sm = sc_feat$sm, indel = sc_feat$indel, norm = "maxlength")
})
write.csv(as.matrix(D_features), file.path(outdir, "ref_om_features.csv"), row.names = TRUE)

# ----- OMtspell (OMspell with tokdep.coeff; TraMineR opt.args) -----
# tokdep.coeff same length as indel; seqdist expands scalar indel to nstates before check
suppressMessages({
  sc_tr <- TraMineR::seqcost(seqdata, method = "TRATE", with.missing = FALSE)
  nstates <- length(states)
  tokdep_coeff <- rep(1, nstates)
  D_omtspell <- TraMineR::seqdist(seqdata, method = "OMspell", sm = sc_tr$sm, indel = sc_tr$indel,
    norm = "YujianBo", expcost = 0.5, opt.args = list(tokdep.coeff = tokdep_coeff))
})
write.csv(as.matrix(D_omtspell), file.path(outdir, "ref_omtspell.csv"), row.names = TRUE)

cat("Reference matrices written to", outdir, "\n")
