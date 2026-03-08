# @Author  : Yapeng Wei
# @File    : seqhmm_reference_posterior.R
# @Desc    :
# R seqHMM reference values for posterior_probs and forward_backward
# consistency testing with sequenzo.seqhmm (Python).
#
# Uses the same synthetic data and parameter configs as
# seqhmm_reference_loglik.R (5 sequences, 8 time points, 3 states).
#
# Usage:
#   Rscript seqhmm_reference_posterior.R [outdir]
#
# Output:
#   ref_posterior_A.csv  - posterior probs for Config A (2 states, basic)
#   ref_posterior_B.csv  - posterior probs for Config B (3 states)
#   ref_posterior_C.csv  - posterior probs for Config C (2 states, sticky)
#   ref_posterior_D.csv  - posterior probs for Config D (2 states, uniform)
#   ref_forward_backward_A.csv - log forward & backward probs for Config A
#
# Requires: seqHMM (>= 1.0.8), TraMineR

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
# Synthetic test data (IDENTICAL to loglik test and Python test)
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

n_sequences <- 5L
n_timepoints <- 8L

# ======================================================================
# Helper: extract posterior probs to a standardized data.frame
# ======================================================================
# Output format: seq_idx (0-based), time_idx (0-based), State_1, State_2, ...
# This matches Python hmmlearn's predict_proba() ordering:
#   rows 0..7 = sequence 0, rows 8..15 = sequence 1, etc.

extract_posterior <- function(hmm_model, n_seq, n_time) {
  post <- as.data.frame(posterior_probs(hmm_model))

  cat("  posterior_probs columns:", paste(names(post), collapse=", "), "\n")
  cat("  first rows:\n")
  print(head(post, 10))

  # seqHMM 2.1.0 returns LONG format: id, time, state, probability
  # We need to pivot to WIDE format: seq_idx, time_idx, State_1, State_2, ...
  states <- sort(unique(post$state))
  n_states <- length(states)
  cat("  unique states:", paste(states, collapse=", "), "\n")

  result_rows <- list()
  row_idx <- 1
  # id column uses integer 1..n_seq (not "s0".."s4")
  unique_ids <- sort(unique(post$id))
  cat("  unique ids:", paste(unique_ids, collapse=", "), "\n")
  for (s_idx in seq_along(unique_ids)) {
    sid <- unique_ids[s_idx]
    seq_rows <- post[post$id == sid, ]
    times <- sort(unique(seq_rows$time))
    for (t_val in times) {
      time_rows <- seq_rows[seq_rows$time == t_val, ]
      row_data <- c(s_idx - 1, t_val - 1)  # 0-based indices
      for (st in states) {
        prob <- time_rows$probability[time_rows$state == st]
        row_data <- c(row_data, prob)
      }
      result_rows[[row_idx]] <- as.numeric(row_data)
      row_idx <- row_idx + 1
    }
  }
  out <- as.data.frame(do.call(rbind, result_rows))
  col_names <- c("seq_idx", "time_idx",
                  paste0("State_", seq_len(n_states)))
  names(out) <- col_names
  return(out)
}

# ======================================================================
# Helper: extract forward_backward to standardized data.frame
# ======================================================================
extract_forward_backward <- function(hmm_model, n_seq, n_time) {
  fb <- as.data.frame(forward_backward(hmm_model))

  cat("  forward_backward columns:", paste(names(fb), collapse=", "), "\n")
  cat("  first rows:\n")
  print(head(fb, 10))

  # seqHMM 2.1.0 returns LONG format: id, time, state, log_alpha, log_beta
  states <- sort(unique(fb$state))
  n_states <- length(states)

  result_rows <- list()
  row_idx <- 1
  unique_ids <- sort(unique(fb$id))
  for (s_idx in seq_along(unique_ids)) {
    sid <- unique_ids[s_idx]
    seq_rows <- fb[fb$id == sid, ]
    times <- sort(unique(seq_rows$time))
    for (t_val in times) {
      time_rows <- seq_rows[seq_rows$time == t_val, ]
      row_data <- c(s_idx - 1, t_val - 1)  # 0-based
      # Forward probs for each state
      for (st in states) {
        val <- time_rows$log_alpha[time_rows$state == st]
        row_data <- c(row_data, val)
      }
      # Backward probs for each state
      for (st in states) {
        val <- time_rows$log_beta[time_rows$state == st]
        row_data <- c(row_data, val)
      }
      result_rows[[row_idx]] <- as.numeric(row_data)
      row_idx <- row_idx + 1
    }
  }
  out <- as.data.frame(do.call(rbind, result_rows))
  col_names <- c("seq_idx", "time_idx",
                  paste0("log_fwd_", seq_len(n_states)),
                  paste0("log_bwd_", seq_len(n_states)))
  names(out) <- col_names
  return(out)
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

post_A <- extract_posterior(hmm_A, n_sequences, n_timepoints)
write.csv(post_A, file.path(outdir, "ref_posterior_A.csv"), row.names = FALSE)
cat("Config A posterior: ", nrow(post_A), "rows,",
    ncol(post_A) - 2, "state columns\n")

fb_A <- extract_forward_backward(hmm_A, n_sequences, n_timepoints)
write.csv(fb_A, file.path(outdir, "ref_forward_backward_A.csv"), row.names = FALSE)
cat("Config A forward_backward:", nrow(fb_A), "rows\n")

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

post_B <- extract_posterior(hmm_B, n_sequences, n_timepoints)
write.csv(post_B, file.path(outdir, "ref_posterior_B.csv"), row.names = FALSE)
cat("Config B posterior: ", nrow(post_B), "rows,",
    ncol(post_B) - 2, "state columns\n")

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

post_C <- extract_posterior(hmm_C, n_sequences, n_timepoints)
write.csv(post_C, file.path(outdir, "ref_posterior_C.csv"), row.names = FALSE)
cat("Config C posterior: ", nrow(post_C), "rows,",
    ncol(post_C) - 2, "state columns\n")

# ======================================================================
# Config D: 2 hidden states, uniform
# ======================================================================
init_D <- c(0.5, 0.5)
trans_D <- matrix(c(0.5, 0.5, 0.5, 0.5), nrow = 2, byrow = TRUE)
emiss_D <- matrix(c(1/3, 1/3, 1/3, 1/3, 1/3, 1/3), nrow = 2, byrow = TRUE)

hmm_D <- build_hmm(observations = seq_obj,
                    transition_probs = trans_D,
                    emission_probs = emiss_D,
                    initial_probs = init_D)

post_D <- extract_posterior(hmm_D, n_sequences, n_timepoints)
write.csv(post_D, file.path(outdir, "ref_posterior_D.csv"), row.names = FALSE)
cat("Config D posterior: ", nrow(post_D), "rows,",
    ncol(post_D) - 2, "state columns\n")

cat("\nAll posterior reference files written to:", outdir, "\n")
