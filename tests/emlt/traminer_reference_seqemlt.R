#!/usr/bin/env Rscript
# Generate TraMineRextras seqemlt reference outputs for parity tests.
# Usage: Rscript tests/emlt/traminer_reference_seqemlt.R

library(TraMineR)
library(TraMineRextras)

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else getwd()
out_dir <- if (length(args) >= 2) args[2] else file.path(root, "tests", "emlt")
data_path <- file.path(root, "sequenzo", "datasets", "dyadic_children.csv")

setwd(root)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
df <- df[1:20, ]

time_cols <- names(df)[sapply(names(df), function(x) {
  clean_x <- sub("^X", "", x)
  num_val <- suppressWarnings(as.numeric(clean_x))
  !is.na(num_val) && clean_x != "" && x != "dyadID" && x != "sex"
})]
time_nums <- sort(as.numeric(sub("^X", "", time_cols)))
time_cols <- as.character(time_nums)
if (any(paste0("X", time_cols) %in% names(df))) {
  time_cols <- paste0("X", time_cols)
}

seqdata <- seqdef(df[, time_cols], alphabet = 1:6, states = as.character(1:6))

write_reference <- function(em, tag) {
  prefix <- file.path(out_dir, paste0("ref_seqemlt_", tag))
  write.csv(em$sit.freq, paste0(prefix, "_sit_freq.csv"))
  write.csv(em$sit.transrate, paste0(prefix, "_sit_transrate.csv"))
  write.csv(em$sit.profil, paste0(prefix, "_sit_profil.csv"))
  write.csv(em$c, paste0(prefix, "_c.csv"))
  write.csv(em$d, paste0(prefix, "_d.csv"))
  write.csv(em$sit.cor, paste0(prefix, "_sit_cor.csv"))
  write.csv(em$coord, paste0(prefix, "_coord.csv"))
  write.csv(em$pca$scores, paste0(prefix, "_pca_scores.csv"))
  write.csv(data.frame(period = em$period, a = 1, b = 1), paste0(prefix, "_meta.csv"), row.names = FALSE)
  cat("Wrote reference files:", tag, "\n")
}

em1 <- seqemlt(seqdata, a = 1, b = 1, weighted = TRUE)
write_reference(em1, "weighted_a1_b1")

em2 <- seqemlt(seqdata, a = 2, b = 3, weighted = FALSE)
write.csv(
  data.frame(a = 2, b = 3, weighted = FALSE),
  file.path(out_dir, "ref_seqemlt_unweighted_a2_b3_meta.csv"),
  row.names = FALSE
)
write_reference(em2, "unweighted_a2_b3")

cat("Done.\n")
