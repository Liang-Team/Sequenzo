#!/usr/bin/env Rscript

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(seqHMM))

rows <- data.frame(
  id = rep(paste0("s", 0:4), each = 4),
  time = rep(1:4, times = 5),
  group = rep(c(-1, -0.5, 0, 0.5, 1), each = 4),
  trend = rep(seq(-1, 1, length.out = 4), times = 5),
  activity = c(
    "A", "B", "C", "A",
    "B", "B", "C", "A",
    "C", "C", "A", "B",
    "A", "C", "B", "B",
    "C", "A", "B", "C"
  )
)
rows$activity <- factor(rows$activity, levels = c("A", "B", "C"))

model <- seqHMM:::build_mnhmm(
  n_states = 3,
  n_clusters = 3,
  emission_formula = activity ~ trend,
  initial_formula = ~ group,
  transition_formula = ~ trend,
  cluster_formula = ~ group,
  data = rows,
  id_var = "id",
  time_var = "time",
  state_names = c("State 1", "State 2", "State 3"),
  cluster_names = c("Cluster 1", "Cluster 2", "Cluster 3"),
  scale = FALSE,
  coefs = list(
    eta_pi = seq(-0.55, 0.55, length.out = 12),
    eta_A = seq(-0.45, 0.45, length.out = 36),
    eta_B = seq(0.50, -0.50, length.out = 36),
    eta_omega = c(0.25, -0.35, 0.45, -0.15)
  )
)

pars <- unlist(model$etas)
records <- list()
for (lambda in c(0, 0.2)) {
  out <- seqHMM:::make_objective_mnhmm(model, lambda = lambda, need_grad = TRUE)(pars)
  records[[length(records) + 1L]] <- data.frame(
    lambda = lambda,
    kind = "objective",
    index = 1L,
    name = "objective",
    value = out$objective
  )
  records[[length(records) + 1L]] <- data.frame(
    lambda = lambda,
    kind = "parameter",
    index = seq_along(pars),
    name = names(pars),
    value = as.numeric(pars)
  )
  records[[length(records) + 1L]] <- data.frame(
    lambda = lambda,
    kind = "gradient",
    index = seq_along(out$gradient),
    name = names(out$gradient),
    value = as.numeric(out$gradient)
  )
}

write.csv(
  do.call(rbind, records),
  file.path(outdir, "ref_mnhmm_objective_gradient.csv"),
  row.names = FALSE
)

cat("Written MNHMM objective/gradient references to", outdir, "\n")
