#!/usr/bin/env Rscript

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(seqHMM))

rows <- data.frame(
  id = rep(paste0("s", 0:5), each = 5),
  time = rep(1:5, times = 6),
  trend = rep(seq(-1, 1, length.out = 5), times = 6),
  activity = c(
    "A", "A", "A", "B", "B",
    "A", "A", "B", "B", "B",
    "A", "A", "A", "A", "B",
    "B", "B", "B", "B", "B",
    "B", "B", "A", "B", "A",
    "A", "B", "B", "A", "B"
  )
)
rows$activity <- factor(rows$activity, levels = c("A", "B"))

initial_probs <- list(
  c(0.50, 0.50),
  c(0.50, 0.50)
)
transition_probs <- list(
  matrix(c(0.55, 0.45, 0.45, 0.55), nrow = 2, byrow = TRUE),
  matrix(c(0.55, 0.45, 0.45, 0.55), nrow = 2, byrow = TRUE)
)
start_eta_B <- rep(0, 8)

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
    eta_B = start_eta_B,
    eta_omega = c(0)
  )
)

base_pars <- unlist(model$etas)
np_pi <- attr(model, "np_pi")
np_A <- attr(model, "np_A")
np_B <- attr(model, "np_B")
eta_B_idx <- np_pi + np_A + seq_len(np_B)
lambda_penalty <- 10.0
objective <- seqHMM:::make_objective_mnhmm(
  model,
  lambda = lambda_penalty,
  need_grad = TRUE
)

objective_eta_B <- function(pars_B) {
  pars <- base_pars
  pars[eta_B_idx] <- pars_B
  objective(pars)$objective
}

gradient_eta_B <- function(pars_B) {
  pars <- base_pars
  pars[eta_B_idx] <- pars_B
  as.numeric(objective(pars)$gradient[eta_B_idx])
}

fit <- optim(
  par = as.numeric(base_pars[eta_B_idx]),
  fn = objective_eta_B,
  gr = gradient_eta_B,
  method = "L-BFGS-B",
  control = list(maxit = 1000, factr = 1e3, pgtol = 1e-8)
)

with_eta_B <- function(pars_B) {
  out <- model
  out$etas$eta_B <- seqHMM:::create_eta_B_mnhmm(
    pars_B,
    out$n_states,
    out$n_symbols,
    seqHMM:::K(out$X_B),
    out$n_clusters
  )
  out$gammas$gamma_B <- split(
    seqHMM:::eta_to_gamma_cube_2d_field(out$etas$eta_B),
    seq_len(out$n_clusters)
  )
  out
}

fitted <- with_eta_B(fit$par)

write.csv(
  data.frame(
    key = c(
      "loglik",
      "objective",
      "convergence",
      "iterations",
      "lambda",
      "n_parameters",
      "max_abs_gradient"
    ),
    value = c(
      as.numeric(logLik(fitted)),
      fit$value,
      fit$convergence,
      fit$counts[["function"]],
      lambda_penalty,
      length(fit$par),
      max(abs(gradient_eta_B(fit$par)))
    )
  ),
  file.path(outdir, "ref_mnhmm_fitted_component_meta.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    index = seq_along(fit$par),
    name = names(base_pars)[eta_B_idx],
    value = as.numeric(fit$par)
  ),
  file.path(outdir, "ref_mnhmm_fitted_component_eta_B.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_emission_probs(fitted)[[1]]),
  file.path(outdir, "ref_mnhmm_fitted_component_emission.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(fitted)),
  file.path(outdir, "ref_mnhmm_fitted_component_posterior_cluster.csv"),
  row.names = FALSE
)

cat("Written fitted MNHMM component-covariate references to", outdir, "\n")
