# @Author  : Yuqi Liang 梁彧祺
# @File    : traminer_reference_omloc_life.R
# @Desc    :
# TraMineR reference distances for OMloc + TRATE and OMloc + CONSTANT on country_life_expectancy_global_deciles.
# Matches RStudio workflow: seqdef(country_df[,-1], id=..., left=NA, gaps=NA, right=NA), seqdist(sm="TRATE", indel="auto").
# Usage: Rscript traminer_reference_omloc_life.R <csv_path> [nrows=0 for all] [outdir=.]
# Writes: ref_omloc_trate_life.csv, ref_omloc_constant_life.csv

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript traminer_reference_omloc_life.R <csv_path> [nrows=0] [outdir=.]")
csv_path <- args[1]
nrows <- if (length(args) >= 2) as.integer(args[2]) else 0L
outdir <- if (length(args) >= 3) args[3] else "."

# Match RStudio exactly: read.csv with defaults (user uses same)
dat <- read.csv(csv_path)
# Match RStudio: country_df[,-1] drops first column (country), id=country_df$country
seq_cols <- dat[, -1, drop = FALSE]
id_vals <- if (nrows > 0 && nrows < nrow(dat)) dat[[1]][seq_len(nrows)] else dat[[1]]
if (nrows > 0 && nrows < nrow(dat)) {
  seq_cols <- seq_cols[seq_len(nrows), , drop = FALSE]
}

# Match RStudio: seqdef(country_df[,-1], id=country_df$country, xtstep=1, left=NA, gaps=NA, right=NA)
# No alphabet - let TraMineR infer (same as user's RStudio)
seqdata <- TraMineR::seqdef(
  seq_cols,
  id = id_vals,
  xtstep = 1,
  left = NA,
  gaps = NA,
  right = NA
)

# ----- OMloc + TRATE (match RStudio: sm="TRATE", indel="auto", with.missing=TRUE) -----
suppressMessages({
  D_omloc_trate <- TraMineR::seqdist(
    seqdata = seqdata,
    method = "OMloc",
    sm = "TRATE",
    indel = "auto",
    with.missing = TRUE
  )
})
write.csv(as.matrix(D_omloc_trate), file.path(outdir, "ref_omloc_trate_life.csv"), row.names = TRUE)

# ----- OMloc + CONSTANT (sm="CONSTANT", indel="auto", with.missing=TRUE) -----
suppressMessages({
  D_omloc_const <- TraMineR::seqdist(
    seqdata = seqdata,
    method = "OMloc",
    sm = "CONSTANT",
    indel = "auto",
    with.missing = TRUE
  )
})
write.csv(as.matrix(D_omloc_const), file.path(outdir, "ref_omloc_constant_life.csv"), row.names = TRUE)

cat("Reference matrices written to", outdir, "\n")
