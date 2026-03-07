# @Author  : Yapeng Wei
# @File    : seqhmm_reference_em.R
# @Desc    :
# R seqHMM reference values for EM algorithm (fit_model) consistency
# testing with sequenzo.seqhmm (Python).
#
# Strategy: start from IDENTICAL initial parameters on both sides,
# run EM to convergence, compare:
#   1. Final log-likelihood
#   2. Converged initial_probs
#   3. Converged transition_probs
#   4. Converged emission_probs
#   5. Number of iterations
#
# Uses the same synthetic data as loglik/posterior/viterbi tests.
#
# Usage:
#   Rscript seqhmm_reference_em.R [outdir]
#
# Output:
#   ref_em_A.csv - EM results for Config A (2-state basic)
#   ref_em_B.csv - EM results for Config B (3-state)
#   ref_em_C.csv - EM results for Config C (2-state sticky)
#
# CSV format: key-value pairs (parameter, value) for easy parsing.
#
# Requires: seqHMM (>= 2.0), TraMineR

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}
if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(TraMineR)
  library(seqHMM)
})

cat("seqHMM version:", as.character(packageVersion("seqHMM")), "\n")

# ======================================================================
# Synthetic test data (IDENTICAL across all test files)
# ======================================================================
seq_data <- data.frame(
  t1 = c("A", "B", "A", "C", "A"),
  t2 = c("A", "C", "B", "C", "A"),
  t3 = c("B", "C", "C", "B", "A"),
  t4 = c("B", "A", "A", "A", "B"),
  t5 = c("C", "A", "B", "A", "B"),
  t6 = c("A", "B", "C", "A", "B"),
  t7 = c("A", "C", "A", "B", "C"),
  t8 = c("B", "C", "B", "C", "C"),
  stringsAsFactors = FALSE
)
seq_obj <- seqdef(seq_data, alphabet = c("A", "B", "C"),
                  id = paste0("s", 0:4))

# ======================================================================
# Helper: run EM and extract all converged parameters
# ======================================================================
run_em_and_extract <- function(hmm_model, config_name, outdir) {
  cat("\n=== Config", config_name, "===\n")

  # Log-likelihood BEFORE fitting
  ll_before <- logLik(hmm_model)
  cat("  logLik before EM:", as.numeric(ll_before), "\n")

  # Run EM with tight convergence
  fitted <- fit_model(
    hmm_model,
    em_step = TRUE,
    global_step = FALSE,
    local_step = FALSE,
    control_em = list(
      maxeval = 1000,
      reltol = 1e-10,
      print_level = 0
    )
  )

  # Debug: show what fit_model returns
  cat("  class(fitted):", class(fitted), "\n")
  cat("  names(fitted):", paste(names(fitted), collapse=", "), "\n")

  # Extract the fitted model
  m <- fitted$model
  cat("  class(m):", class(m), "\n")
  cat("  names(m):", paste(names(m), collapse=", "), "\n")

  # Converged parameters -- try accessor functions first, fall back to list access
  ip <- tryCatch(initial_probs(m), error = function(e) m$initial_probs)
  tp <- tryCatch(transition_probs(m), error = function(e) m$transition_probs)
  ep <- tryCatch(emission_probs(m), error = function(e) m$emission_probs)
  ll_after <- as.numeric(logLik(m))

  cat("  logLik after EM:", ll_after, "\n")
  cat("  class(ip):", class(ip), "\n")
  cat("  initial_probs:", ip, "\n")
  cat("  class(tp):", class(tp), "\n")
  cat("  transition_probs:\n")
  print(tp)
  cat("  class(ep):", class(ep), "\n")
  cat("  emission_probs:\n")
  print(ep)

  # Build output as a flat key-value CSV
  # This makes it easy to parse in Python without worrying about matrix formats
  rows <- list()
  idx <- 1

  # Log-likelihood
  rows[[idx]] <- data.frame(key = "loglik_before", value = as.numeric(ll_before))
  idx <- idx + 1
  rows[[idx]] <- data.frame(key = "loglik_after", value = ll_after)
  idx <- idx + 1

  # Initial probs (may be named vector)
  ip_vec <- as.numeric(ip)
  for (i in seq_along(ip_vec)) {
    rows[[idx]] <- data.frame(
      key = paste0("initial_probs_", i - 1),
      value = ip_vec[i]
    )
    idx <- idx + 1
  }

  # Transition probs (row-major: tp[i,j] -> trans_i_j)
  tp_mat <- as.matrix(tp)
  n_states <- nrow(tp_mat)
  for (i in 1:n_states) {
    for (j in 1:n_states) {
      rows[[idx]] <- data.frame(
        key = paste0("trans_", i - 1, "_", j - 1),
        value = tp_mat[i, j]
      )
      idx <- idx + 1
    }
  }

  # Emission probs -- may be a list (one per channel) or a matrix
  if (is.list(ep) && !is.matrix(ep)) {
    ep_mat <- as.matrix(ep[[1]])  # single channel: first element
  } else {
    ep_mat <- as.matrix(ep)
  }
  n_symbols <- ncol(ep_mat)
  for (i in 1:n_states) {
    for (j in 1:n_symbols) {
      rows[[idx]] <- data.frame(
        key = paste0("emiss_", i - 1, "_", j - 1),
        value = ep_mat[i, j]
      )
      idx <- idx + 1
    }
  }

  out <- do.call(rbind, rows)
  fname <- file.path(outdir, paste0("ref_em_", config_name, ".csv"))
  write.csv(out, fname, row.names = FALSE)
  cat("  Written:", fname, "(", nrow(out), "rows)\n")
}

# ======================================================================
# Config A: 2 hidden states, basic
# ======================================================================
init_A <- c(0.6, 0.4)
trans_A <- matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_A <- matrix(c(0.5, 0.3, 0.2, 0.1, 0.4, 0.5), nrow = 2, byrow = TRUE)

hmm_A <- build_hmm(observations = seq_obj,
                    transition_probs = trans_A,
                    emission_probs = emiss_A,
                    initial_probs = init_A)
run_em_and_extract(hmm_A, "A", outdir)

# ======================================================================
# Config B: 3 hidden states
# ======================================================================
init_B <- c(0.5, 0.3, 0.2)
trans_B <- matrix(c(0.7, 0.2, 0.1, 0.1, 0.7, 0.2, 0.2, 0.1, 0.7),
                  nrow = 3, byrow = TRUE)
emiss_B <- matrix(c(0.6, 0.3, 0.1, 0.1, 0.6, 0.3, 0.3, 0.1, 0.6),
                  nrow = 3, byrow = TRUE)

hmm_B <- build_hmm(observations = seq_obj,
                    transition_probs = trans_B,
                    emission_probs = emiss_B,
                    initial_probs = init_B)
run_em_and_extract(hmm_B, "B", outdir)

# ======================================================================
# Config C: 2 hidden states, sticky
# ======================================================================
init_C <- c(0.9, 0.1)
trans_C <- matrix(c(0.95, 0.05, 0.05, 0.95), nrow = 2, byrow = TRUE)
emiss_C <- matrix(c(0.8, 0.15, 0.05, 0.05, 0.15, 0.8), nrow = 2, byrow = TRUE)

hmm_C <- build_hmm(observations = seq_obj,
                    transition_probs = trans_C,
                    emission_probs = emiss_C,
                    initial_probs = init_C)
run_em_and_extract(hmm_C, "C", outdir)

cat("\nAll EM reference files written to:", outdir, "\n")
