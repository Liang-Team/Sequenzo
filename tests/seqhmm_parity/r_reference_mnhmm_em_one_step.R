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
  activity = c(
    "A", "A", "A", "A", "A",
    "A", "A", "B", "A", "A",
    "B", "B", "B", "B", "B",
    "B", "B", "A", "B", "B"
  )
)
rows$activity <- factor(rows$activity, levels = c("A", "B"))

coefs <- list(
  initial_probs = list(
    c(0.95, 0.05),
    c(0.05, 0.95)
  ),
  transition_probs = list(
    matrix(c(0.98, 0.02, 0.10, 0.90), nrow = 2, byrow = TRUE),
    matrix(c(0.90, 0.10, 0.02, 0.98), nrow = 2, byrow = TRUE)
  ),
  emission_probs = list(
    matrix(c(0.98, 0.02, 0.30, 0.70), nrow = 2, byrow = TRUE),
    matrix(c(0.70, 0.30, 0.02, 0.98), nrow = 2, byrow = TRUE)
  ),
  cluster_probs = c(0.5, 0.5)
)

model <- seqHMM:::build_mnhmm(
  n_states = 2,
  n_clusters = 2,
  emission_formula = activity ~ 1,
  initial_formula = ~ 1,
  transition_formula = ~ 1,
  cluster_formula = ~ 1,
  data = rows,
  id_var = "id",
  time_var = "time",
  state_names = c("State 1", "State 2"),
  cluster_names = c("Cluster 1", "Cluster 2"),
  scale = FALSE,
  coefs = coefs
)

fit <- seqHMM:::fit_mnhmm(
  model = model,
  inits = coefs,
  init_sd = 0,
  restarts = 0L,
  lambda = 0,
  method = "EM",
  bound = Inf,
  control = list(maxeval = 1, use_squarem = FALSE, print_level = 0),
  control_restart = list(),
  control_mstep = list(maxeval = 100, print_level = 0)
)

write.csv(
  data.frame(
    key = c("loglik", "iterations", "return_code"),
    value = c(
      as.numeric(logLik(fit)),
      fit$estimation_results$iterations,
      fit$estimation_results$return_code
    )
  ),
  file.path(outdir, "ref_mnhmm_em_one_step_meta.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_initial_probs(fit)),
  file.path(outdir, "ref_mnhmm_em_one_step_initial.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_transition_probs(fit)),
  file.path(outdir, "ref_mnhmm_em_one_step_transition.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_emission_probs(fit)[[1]]),
  file.path(outdir, "ref_mnhmm_em_one_step_emission.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_cluster_probs(fit)),
  file.path(outdir, "ref_mnhmm_em_one_step_prior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(fit)),
  file.path(outdir, "ref_mnhmm_em_one_step_posterior_cluster.csv"),
  row.names = FALSE
)

cat("Written MNHMM one-step EM references to", outdir, "\n")
