# @Author  : Yuqi Liang 梁彧祺
# @File    : traminer_reference.R
# @Time    : 2026/02/08 13:14
# @Desc    :
# TraMineR reference distances for: OM+INDELS, OM+INDELSLOG, OM+FUTURE, OM+FEATURES, OMtspell, OMstran.
# Usage: Rscript traminer_reference.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example: Rscript traminer_reference.R ../../../sequenzo/datasets/dyadic_children.csv 10 .
# Writes: ref_om_indels.csv, ref_om_indelslog.csv, ref_om_future.csv, ref_om_features.csv,
#         ref_omtspell.csv, ref_omstran*.csv

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
nstates <- length(states)
tokdep_coeff <- rep(1, nstates)
suppressMessages({
  D_omtspell <- TraMineR::seqdist(seqdata, method = "OMspell", sm = "TRATE", indel = "auto",
    norm = "YujianBo", expcost = 0.5, opt.args = list(tokdep.coeff = tokdep_coeff))
})
write.csv(as.matrix(D_omtspell), file.path(outdir, "ref_omtspell.csv"), row.names = TRUE)

# ----- Part 1b: OM parameter configs (for TraMineR comparison) -----
# OM + INDELS, norm = "none"
suppressMessages({
  D_indels_none <- TraMineR::seqdist(seqdata, method = "OM", sm = "INDELS", indel = "auto", norm = "none")
})
write.csv(as.matrix(D_indels_none), file.path(outdir, "ref_om_indels_norm_none.csv"), row.names = TRUE)

# OM + INDELS, indel = 1 (scalar), norm = "maxlength"
suppressMessages({
  D_indels_indel1 <- TraMineR::seqdist(seqdata, method = "OM", sm = "INDELS", indel = 1, norm = "maxlength")
})
write.csv(as.matrix(D_indels_indel1), file.path(outdir, "ref_om_indels_indel1_maxlength.csv"), row.names = TRUE)

# OM + TRATE, norm = "gmean"
suppressMessages({
  D_trate_gmean <- TraMineR::seqdist(seqdata, method = "OM", sm = "TRATE", indel = "auto", norm = "gmean")
})
write.csv(as.matrix(D_trate_gmean), file.path(outdir, "ref_om_trate_gmean.csv"), row.names = TRUE)

# OMtspell with expcost = 0.3 and 0.7
for (expcost_val in c(0.3, 0.7)) {
  suppressMessages({
    D_ec <- TraMineR::seqdist(seqdata, method = "OMspell", sm = "TRATE", indel = "auto",
      norm = "YujianBo", expcost = expcost_val, opt.args = list(tokdep.coeff = tokdep_coeff))
  })
  write.csv(as.matrix(D_ec), file.path(outdir, sprintf("ref_omtspell_expcost%02.0f.csv", expcost_val * 10)), row.names = TRUE)
}

# ----- Part 1c: OMstran (OM of transition sequences) -----
# OMstran: sm=TRATE, indel=auto, transindel=constant, otto=0.5, previous=FALSE, add.column=TRUE, norm=YujianBo
suppressMessages({
  D_omstran <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.5, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran), file.path(outdir, "ref_omstran.csv"), row.names = TRUE)

# OMstran transindel=prob
suppressMessages({
  D_omstran_prob <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "prob", otto = 0.5, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_prob), file.path(outdir, "ref_omstran_prob.csv"), row.names = TRUE)

# OMstran transindel=subcost
suppressMessages({
  D_omstran_subcost <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "subcost", otto = 0.5, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_subcost), file.path(outdir, "ref_omstran_subcost.csv"), row.names = TRUE)

# OMstran otto=0.3
suppressMessages({
  D_omstran_otto03 <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.3, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_otto03), file.path(outdir, "ref_omstran_otto03.csv"), row.names = TRUE)

# OMstran otto=0.7
suppressMessages({
  D_omstran_otto07 <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.7, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_otto07), file.path(outdir, "ref_omstran_otto07.csv"), row.names = TRUE)

# OMstran previous=TRUE
suppressMessages({
  D_omstran_previous <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.5, previous = TRUE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_previous), file.path(outdir, "ref_omstran_previous.csv"), row.names = TRUE)

# OMstran add.column=FALSE
suppressMessages({
  D_omstran_addcol_false <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.5, previous = FALSE, add.column = FALSE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_addcol_false), file.path(outdir, "ref_omstran_addcolumn_false.csv"), row.names = TRUE)

# OMstran norm=none
suppressMessages({
  D_omstran_norm_none <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = "auto",
    transindel = "constant", otto = 0.5, previous = FALSE, add.column = TRUE, norm = "none")
})
write.csv(as.matrix(D_omstran_norm_none), file.path(outdir, "ref_omstran_norm_none.csv"), row.names = TRUE)

# OMstran indel=1 (scalar)
suppressMessages({
  D_omstran_indel1 <- TraMineR::seqdist(seqdata, method = "OMstran", sm = "TRATE", indel = 1,
    transindel = "constant", otto = 0.5, previous = FALSE, add.column = TRUE, norm = "YujianBo")
})
write.csv(as.matrix(D_omstran_indel1), file.path(outdir, "ref_omstran_indel1.csv"), row.names = TRUE)

cat("Reference matrices written to", outdir, "\n")
