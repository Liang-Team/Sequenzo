#!/usr/bin/env Rscript
# WeightedCluster reference outputs for seqpropclust / property extraction.
# Usage: Rscript tests/clustering/weightedcluster_reference_seqpropclust.R [outdir]

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(TraMineR)
  library(WeightedCluster)
})

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_props <- function(props, filename) {
  write.csv(props, file.path(outdir, filename), row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Tiny panel (matches tests/clustering/test_property_based_clustering.py)
# ---------------------------------------------------------------------------
tiny_df <- data.frame(
  id = 1:3,
  `1` = c(1, 1, 2),
  `2` = c(1, 2, 2),
  `3` = c(2, 2, 3),
  `4` = c(2, 3, 3),
  check.names = FALSE
)
tiny_seq <- seqdef(
  tiny_df[, c("1", "2", "3", "4")],
  alphabet = 1:3,
  states = c("A", "B", "C"),
  labels = c("A", "B", "C")
)

tiny_state <- seqpropclust(tiny_seq, diss = NULL, properties = c("state"), prop.only = TRUE)
tiny_duration <- seqpropclust(tiny_seq, diss = NULL, properties = c("duration"), prop.only = TRUE)
tiny_spell <- seqpropclust(
  tiny_seq,
  diss = NULL,
  properties = c("spell.age", "spell.dur"),
  prop.only = TRUE
)
tiny_all <- seqpropclust(
  tiny_seq,
  diss = NULL,
  properties = c("state", "duration", "spell.age", "spell.dur"),
  prop.only = TRUE
)

write_props(tiny_state, "ref_seqpropclust_tiny_state.csv")
write_props(tiny_duration, "ref_seqpropclust_tiny_duration.csv")
write_props(tiny_spell, "ref_seqpropclust_tiny_spell.csv")
write_props(tiny_all, "ref_seqpropclust_tiny_all.csv")

# ---------------------------------------------------------------------------
# lsog subset (first 20 rows, same as discrepancy tests)
# ---------------------------------------------------------------------------
project_root <- normalizePath(file.path(outdir, "..", ".."), mustWork = FALSE)
lsog_path <- file.path(project_root, "sequenzo", "datasets", "dyadic_children.csv")
if (!file.exists(lsog_path)) {
  stop("Could not find dyadic_children.csv at ", lsog_path)
}

lsog_df <- read.csv(lsog_path, stringsAsFactors = FALSE, check.names = FALSE)
lsog_df <- lsog_df[1:20, ]
time_cols <- names(lsog_df)[sapply(names(lsog_df), function(x) {
  clean_x <- sub("^X", "", x)
  num_val <- suppressWarnings(as.numeric(clean_x))
  !is.na(num_val) && clean_x != "" && x != "dyadID" && x != "sex"
})]
time_nums <- sort(as.numeric(sub("^X", "", time_cols)))
time_cols <- as.character(time_nums)
if (any(paste0("X", time_cols) %in% names(lsog_df))) {
  time_cols_r <- paste0("X", time_cols)
} else {
  time_cols_r <- time_cols
}

lsog_seq <- seqdef(
  lsog_df[, time_cols_r],
  alphabet = 1:6,
  states = as.character(1:6)
)

lsog_props <- seqpropclust(
  lsog_seq,
  diss = NULL,
  properties = c("state", "duration", "spell.age", "spell.dur", "Complexity"),
  prop.only = TRUE
)
write_props(lsog_props, "ref_seqpropclust_lsog_core.csv")

lsog_diss <- as.matrix(seqdist(lsog_seq, method = "LCS", norm = "auto"))
write.csv(lsog_diss, file.path(outdir, "ref_seqpropclust_lsog_lcs.csv"), row.names = FALSE)

lsog_tree <- seqpropclust(
  lsog_seq,
  diss = lsog_diss,
  properties = c("state", "duration"),
  maxcluster = 6,
  R = 1,
  weight.permutation = "diss",
  min.size = 0.01,
  max.depth = 5,
  pval = 1.0
)

write.csv(
  data.frame(id = lsog_df$dyadID, fitted = lsog_tree$fitted[, 1]),
  file.path(outdir, "ref_seqpropclust_lsog_fitted.csv"),
  row.names = FALSE
)

cuts <- data.frame(id = lsog_df$dyadID)
for (k in 2:6) {
  cuts[[paste0("Split", k)]] <- as.character(dtcut(lsog_tree, k, labels = FALSE))
}
write.csv(cuts, file.path(outdir, "ref_seqpropclust_lsog_cuts.csv"), row.names = FALSE)

schedules <- data.frame(
  node_id = integer(),
  splitschedule = integer()
)
collect_schedule <- function(node) {
  schedules <<- rbind(
    schedules,
    data.frame(node_id = node$id, splitschedule = node$info$splitschedule)
  )
  if (!is.null(node$kids)) {
    collect_schedule(node$kids[[1]])
    collect_schedule(node$kids[[2]])
  }
}
collect_schedule(lsog_tree$root)
write.csv(schedules, file.path(outdir, "ref_seqpropclust_lsog_schedules.csv"), row.names = FALSE)

cat("Wrote seqpropclust reference files to", outdir, "\n")
