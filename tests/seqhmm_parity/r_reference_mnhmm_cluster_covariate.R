#!/usr/bin/env Rscript

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(seqHMM))

rows <- data.frame(
  id = rep(paste0("s", 0:3), each = 5),
  time = rep(1:5, times = 4),
  group = rep(c(0, 0, 1, 1), each = 5),
  activity = c(
    "A", "A", "A", "A", "A",
    "A", "A", "B", "A", "A",
    "B", "B", "B", "B", "B",
    "B", "B", "A", "B", "B"
  )
)
rows$activity <- factor(rows$activity, levels = c("A", "B"))

initial_probs <- list(
  c(0.95, 0.05),
  c(0.05, 0.95)
)
transition_probs <- list(
  matrix(c(0.98, 0.02, 0.10, 0.90), nrow = 2, byrow = TRUE),
  matrix(c(0.90, 0.10, 0.02, 0.98), nrow = 2, byrow = TRUE)
)
emission_probs <- list(
  matrix(c(0.98, 0.02, 0.30, 0.70), nrow = 2, byrow = TRUE),
  matrix(c(0.70, 0.30, 0.02, 0.98), nrow = 2, byrow = TRUE)
)

model <- seqHMM:::build_mnhmm(
  n_states = 2,
  n_clusters = 2,
  emission_formula = activity ~ 1,
  initial_formula = ~ 1,
  transition_formula = ~ 1,
  cluster_formula = ~ group,
  data = rows,
  id_var = "id",
  time_var = "time",
  scale = FALSE,
  coefs = list(
    initial_probs = initial_probs,
    transition_probs = transition_probs,
    emission_probs = emission_probs,
    eta_omega = c(1, -1)
  )
)

write.csv(
  data.frame(key = "loglik", value = as.numeric(logLik(model))),
  file.path(outdir, "ref_mnhmm_cov_loglik.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_cluster_probs(model)),
  file.path(outdir, "ref_mnhmm_cov_prior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(model)),
  file.path(outdir, "ref_mnhmm_cov_posterior_cluster.csv"),
  row.names = FALSE
)

cat("Written MNHMM covariate references to", outdir, "\n")
