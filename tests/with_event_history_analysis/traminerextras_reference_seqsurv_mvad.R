#!/usr/bin/env Rscript
# Reference output for TraMineRextras seqsurv on mvad (spell survival by state).
# Usage: Rscript tests/with_event_history_analysis/traminerextras_reference_seqsurv_mvad.R [outdir]

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else file.path("tests", "with_event_history_analysis", "reference_data")

suppressPackageStartupMessages({
  library(TraMineR)
  library(TraMineRextras)
})

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

data(mvad)
alphabet <- c("employment", "FE", "HE", "joblessness", "school", "training")
lab <- c(
  "employment", "further education", "higher education",
  "joblessness", "school", "training"
)
mvad.seq <- seqdef(mvad, 17:86, alphabet = alphabet, labels = lab, xtstep = 6)
res <- seqsurv(mvad.seq)

# ``res$strata`` holds *counts* per stratum, not cumulative end indices.
counts <- as.integer(res$strata)
ends <- cumsum(counts)
starts <- c(1L, head(ends, -1L) + 1L)
strata_names <- names(res$strata)

out <- list()
for (i in seq_along(starts)) {
  a <- starts[i]
  b <- ends[i]
  nm <- sub("^spell\\$states=", "", strata_names[i])
  out[[i]] <- data.frame(
    strata = nm,
    time = res$time[a:b],
    n.risk = res$n.risk[a:b],
    n.event = res$n.event[a:b],
    surv = res$surv[a:b],
    std.err = res$std.err[a:b]
  )
}
df <- do.call(rbind, out)
write.csv(df, file.path(outdir, "ref_mvad_seqsurv.csv"), row.names = FALSE)

cat("Wrote", nrow(df), "rows to", outdir, "\n")
