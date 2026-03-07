# @Author  : Yapeng Wei
# @File    : seqhmm_reference_viterbi.R
# @Desc    :
# R seqHMM reference values for hidden_paths (Viterbi algorithm)
# consistency testing with sequenzo.seqhmm (Python).
#
# Uses the same synthetic data and parameter configs as the loglik
# and posterior tests.
#
# Usage:
#   Rscript seqhmm_reference_viterbi.R [outdir]
#
# Output:
#   ref_viterbi_A.csv  - Viterbi paths for Config A
#   ref_viterbi_B.csv  - Viterbi paths for Config B
#   ref_viterbi_C.csv  - Viterbi paths for Config C
#   ref_viterbi_D.csv  - Viterbi paths for Config D
#
# CSV format:
#   seq_idx (0-based), time_idx (0-based), state_idx (0-based integer)
#   Plus a separate column for per-sequence Viterbi log-probability.
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

n_sequences <- 5L
n_timepoints <- 8L

# ======================================================================
# Helper: extract hidden_paths to standardized CSV
# ======================================================================
# seqHMM 2.1.0 hidden_paths() returns:
#   - data.table with columns: id (int), time (int), state (char)
#   - attr(result, "log_prob"): named numeric vector of per-sequence
#     Viterbi log-probabilities
#
# We convert state names "State 1", "State 2", ... to 0-based indices
# and output: seq_idx, time_idx, state_idx, viterbi_logprob

extract_viterbi <- function(hmm_model, n_seq, n_time) {
  hp <- hidden_paths(hmm_model)
  log_probs <- attr(hp, "log_prob")
  hp <- as.data.frame(hp)

  cat("  hidden_paths columns:", paste(names(hp), collapse=", "), "\n")
  cat("  first rows:\n")
  print(head(hp, 10))

  # Build state name -> 0-based index mapping
  states <- sort(unique(hp$state))
  cat("  unique states:", paste(states, collapse=", "), "\n")
  state_map <- setNames(seq_along(states) - 1L, states)

  # Extract per-sequence Viterbi log-probabilities
  cat("  log_prob attr:", paste(log_probs, collapse=", "), "\n")

  unique_ids <- sort(unique(hp$id))
  cat("  unique ids:", paste(unique_ids, collapse=", "), "\n")

  result_rows <- list()
  row_idx <- 1
  for (s_idx in seq_along(unique_ids)) {
    sid <- unique_ids[s_idx]
    seq_rows <- hp[hp$id == sid, ]
    times <- sort(unique(seq_rows$time))

    # Per-sequence Viterbi log-prob
    vit_lp <- log_probs[s_idx]

    for (t_val in times) {
      time_row <- seq_rows[seq_rows$time == t_val, ]
      st_name <- time_row$state[1]
      st_idx <- state_map[st_name]
      result_rows[[row_idx]] <- c(s_idx - 1, t_val - 1, st_idx, vit_lp)
      row_idx <- row_idx + 1
    }
  }
  out <- as.data.frame(do.call(rbind, result_rows))
  names(out) <- c("seq_idx", "time_idx", "state_idx", "viterbi_logprob")
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

vit_A <- extract_viterbi(hmm_A, n_sequences, n_timepoints)
write.csv(vit_A, file.path(outdir, "ref_viterbi_A.csv"), row.names = FALSE)
cat("Config A:", nrow(vit_A), "rows written\n\n")

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

vit_B <- extract_viterbi(hmm_B, n_sequences, n_timepoints)
write.csv(vit_B, file.path(outdir, "ref_viterbi_B.csv"), row.names = FALSE)
cat("Config B:", nrow(vit_B), "rows written\n\n")

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

vit_C <- extract_viterbi(hmm_C, n_sequences, n_timepoints)
write.csv(vit_C, file.path(outdir, "ref_viterbi_C.csv"), row.names = FALSE)
cat("Config C:", nrow(vit_C), "rows written\n\n")

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

vit_D <- extract_viterbi(hmm_D, n_sequences, n_timepoints)
write.csv(vit_D, file.path(outdir, "ref_viterbi_D.csv"), row.names = FALSE)
cat("Config D:", nrow(vit_D), "rows written\n\n")

cat("All Viterbi reference files written to:", outdir, "\n")
