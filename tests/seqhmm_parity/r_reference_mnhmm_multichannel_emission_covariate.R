#!/usr/bin/env Rscript

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(seqHMM))

rows <- data.frame(
  id = rep(paste0("s", 0:3), each = 4),
  time = rep(1:4, times = 4),
  trend = rep(seq(-1.0, 1.0, length.out = 4), times = 4),
  ch1 = c(
    "A", "A", "A", "A",
    "A", "B", "A", "A",
    "B", "B", "B", "B",
    "B", "A", "B", "B"
  ),
  ch2 = c(
    "X", "X", "Y", "X",
    "X", "Y", "Y", "X",
    "Y", "Y", "Y", "Y",
    "Y", "X", "X", "Y"
  )
)
rows$ch1 <- factor(rows$ch1, levels = c("A", "B"))
rows$ch2 <- factor(rows$ch2, levels = c("X", "Y"))

initial_probs <- list(
  c(0.90, 0.10),
  c(0.10, 0.90)
)
transition_probs <- list(
  matrix(c(0.85, 0.15, 0.25, 0.75), nrow = 2, byrow = TRUE),
  matrix(c(0.70, 0.30, 0.10, 0.90), nrow = 2, byrow = TRUE)
)
eta_B <- list(
  list(
    array(c(0.30, 0.20, -0.10, -0.25), dim = c(1, 2, 2)),
    array(c(0.10, -0.15, 0.25, 0.05), dim = c(1, 2, 2))
  ),
  list(
    array(c(-0.20, 0.15, 0.35, -0.05), dim = c(1, 2, 2)),
    array(c(0.25, -0.10, -0.30, 0.20), dim = c(1, 2, 2))
  )
)

model <- seqHMM:::build_mnhmm(
  n_states = 2,
  n_clusters = 2,
  emission_formula = c(ch1, ch2) ~ trend,
  initial_formula = ~ 1,
  transition_formula = ~ 1,
  cluster_formula = ~ 1,
  data = rows,
  id_var = "id",
  time_var = "time",
  scale = FALSE,
  state_names = c("State 1", "State 2"),
  coefs = list(
    initial_probs = initial_probs,
    transition_probs = transition_probs,
    eta_B = eta_B,
    cluster_probs = c(0.55, 0.45)
  )
)

write.csv(
  data.frame(key = "loglik", value = as.numeric(logLik(model))),
  file.path(outdir, "ref_mnhmm_multichannel_emission_cov_loglik.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(posterior_cluster_probabilities(model)),
  file.path(outdir, "ref_mnhmm_multichannel_emission_cov_posterior_cluster.csv"),
  row.names = FALSE
)

emission <- Map(
  function(channel_idx, frame) {
    frame <- as.data.frame(frame)
    response <- names(get_emission_probs(model))[channel_idx]
    data.frame(
      channel = paste("Channel", channel_idx),
      cluster = frame$cluster,
      id = frame$id,
      time = frame$time,
      state = frame$state,
      symbol = frame[[response]],
      probability = frame$probability
    )
  },
  seq_along(get_emission_probs(model)),
  get_emission_probs(model)
)
write.csv(
  do.call(rbind, emission),
  file.path(outdir, "ref_mnhmm_multichannel_emission_cov_emission.csv"),
  row.names = FALSE
)

cat("Written multichannel MNHMM emission-covariate references to", outdir, "\n")
