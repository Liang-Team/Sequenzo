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
  group = rep(c(0, 1, 1, 0), each = 5),
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
    eta_omega = c(0, 0)
  )
)

with_eta_omega <- function(pars) {
  out <- model
  out$etas$eta_omega <- matrix(pars, nrow = 1)
  out$gammas$gamma_omega <- seqHMM:::eta_to_gamma_mat(out$etas$eta_omega)
  out
}

objective <- function(pars) {
  -as.numeric(logLik(with_eta_omega(pars)))
}

gradient <- function(pars) {
  fitted <- with_eta_omega(pars)
  priors <- as.data.frame(get_cluster_probs(fitted))
  posteriors <- as.data.frame(posterior_cluster_probabilities(fitted))
  prior_mat <- as.matrix(
    reshape(
      priors,
      idvar = "id",
      timevar = "cluster",
      direction = "wide"
    )[paste0("probability.Cluster ", 1:2)]
  )
  posterior_mat <- as.matrix(
    reshape(
      posteriors,
      idvar = "id",
      timevar = "cluster",
      direction = "wide"
    )[paste0("probability.Cluster ", 1:2)]
  )
  X <- unique(rows[c("id", "group")])
  X <- as.matrix(cbind(`(Intercept)` = 1, group = X$group))
  grad_full <- t(X) %*% (posterior_mat - prior_mat)
  -as.numeric(t(seqHMM:::create_Q(2)) %*% t(grad_full))
}

fit <- optim(
  par = c(0, 0),
  fn = objective,
  gr = gradient,
  method = "BFGS",
  control = list(maxit = 1000, reltol = 1e-14)
)

fitted <- with_eta_omega(fit$par)

write.csv(
  data.frame(
    key = c("loglik", "convergence", "objective"),
    value = c(as.numeric(logLik(fitted)), fit$convergence, fit$value)
  ),
  file.path(outdir, "ref_mnhmm_fitted_cov_meta.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(get_cluster_probs(fitted)),
  file.path(outdir, "ref_mnhmm_fitted_cov_prior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(fitted)),
  file.path(outdir, "ref_mnhmm_fitted_cov_posterior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    covariate = c("(Intercept)", "group"),
    value = as.numeric(fit$par)
  ),
  file.path(outdir, "ref_mnhmm_fitted_cov_eta_omega.csv"),
  row.names = FALSE
)

cat("Written fitted MNHMM cluster covariate references to", outdir, "\n")
