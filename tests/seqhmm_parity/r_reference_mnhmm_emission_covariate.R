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
  trend = rep(seq(-1, 1, length.out = 5), times = 4),
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

model <- seqHMM:::build_mnhmm(
  n_states = 2,
  n_clusters = 2,
  emission_formula = activity ~ trend,
  initial_formula = ~ 1,
  transition_formula = ~ 1,
  cluster_formula = ~ 1,
  data = rows,
  id_var = "id",
  time_var = "time",
  state_names = c("State 1", "State 2"),
  cluster_names = c("Cluster 1", "Cluster 2"),
  scale = FALSE,
  coefs = list(
    initial_probs = initial_probs,
    transition_probs = transition_probs,
    eta_B = c(0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5),
    eta_omega = c(0)
  )
)

write.csv(
  data.frame(key = "loglik", value = as.numeric(logLik(model))),
  file.path(outdir, "ref_mnhmm_emission_cov_loglik.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(model)),
  file.path(outdir, "ref_mnhmm_emission_cov_posterior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_emission_probs(model)[[1]]),
  file.path(outdir, "ref_mnhmm_emission_cov_emission.csv"),
  row.names = FALSE
)

cat("Written MNHMM emission covariate references to", outdir, "\n")
