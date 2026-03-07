# @Author  : Yapeng Wei
# @File    : seqhmm_reference_simulate.R
# @Desc    :
# R seqHMM reference values for simulate_hmm / simulate_mhmm
# consistency testing with sequenzo.seqhmm (Python).
#
# Strategy: Given fixed parameters and seeds, simulate sequences in R,
# then verify that Python produces identical sequences (same RNG logic)
# OR verify statistical properties (state frequencies, transition counts).
#
# Since R and Python use different RNGs, exact sequence matching is
# unlikely. Instead we test STATISTICAL PROPERTIES:
#   1. Empirical state frequencies converge to stationary distribution
#   2. Empirical transition counts match expected transition probs
#   3. Simulated sequence dimensions are correct
#   4. With enough samples, marginal distributions match
#
# For reproducibility testing, we record R's exact output with set.seed().
#
# Usage:
#   Rscript seqhmm_reference_simulate.R [outdir]
#
# Output:
#   ref_sim_hmm_sequences.csv  - Simulated sequences from HMM
#   ref_sim_hmm_states.csv     - Simulated hidden states from HMM
#   ref_sim_hmm_stats.csv      - Summary statistics
#   ref_sim_mhmm_stats.csv     - Summary statistics for MHMM simulation
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
# HMM Simulation: 2 states, 3 symbols
# ======================================================================
init_sim <- c(0.6, 0.4)
trans_sim <- matrix(c(0.7, 0.3,
                      0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_sim <- matrix(c(0.5, 0.3, 0.2,
                       0.1, 0.4, 0.5), nrow = 2, byrow = TRUE)
colnames(emiss_sim) <- c("A", "B", "C")

n_seq <- 100
seq_len <- 20

set.seed(42)
sim_hmm <- simulate_hmm(
  n_sequences = n_seq,
  initial_probs = init_sim,
  transition_probs = trans_sim,
  emission_probs = emiss_sim,
  sequence_length = seq_len
)

cat("\nclass(sim_hmm):", class(sim_hmm), "\n")
cat("names(sim_hmm):", paste(names(sim_hmm), collapse = ", "), "\n")

# Extract observed sequences
obs <- sim_hmm$observations
obs_mat <- as.matrix(obs)
cat("Observed sequences dim:", dim(obs_mat), "\n")

# Extract hidden states
states <- sim_hmm$states
states_mat <- as.matrix(states)
cat("Hidden states dim:", dim(states_mat), "\n")

# Save raw sequences (use 1:nrow for IDs since rownames may be NULL in seqHMM 2.x)
write.csv(
  data.frame(id = 1:nrow(obs_mat), obs_mat, stringsAsFactors = FALSE),
  file.path(outdir, "ref_sim_hmm_sequences.csv"),
  row.names = FALSE
)
cat("Written: ref_sim_hmm_sequences.csv\n")

write.csv(
  data.frame(id = 1:nrow(states_mat), states_mat, stringsAsFactors = FALSE),
  file.path(outdir, "ref_sim_hmm_states.csv"),
  row.names = FALSE
)
cat("Written: ref_sim_hmm_states.csv\n")

# Compute summary statistics for statistical comparison
# 1. Overall symbol frequencies
obs_vec <- as.vector(obs_mat)
symbol_freq <- table(obs_vec) / length(obs_vec)
cat("\nSymbol frequencies:\n")
print(symbol_freq)

# 2. State frequencies
states_vec <- as.vector(states_mat)
state_freq <- table(states_vec) / length(states_vec)
cat("\nState frequencies:\n")
print(state_freq)

# 3. Empirical transition counts in hidden states
n_states <- nrow(trans_sim)
state_labels <- sort(unique(states_vec))
emp_trans <- matrix(0, nrow = n_states, ncol = n_states)
for (i in 1:nrow(states_mat)) {
  for (t in 1:(ncol(states_mat) - 1)) {
    from <- match(states_mat[i, t], state_labels)
    to <- match(states_mat[i, t + 1], state_labels)
    if (!is.na(from) && !is.na(to)) {
      emp_trans[from, to] <- emp_trans[from, to] + 1
    }
  }
}
# Normalize to probabilities
emp_trans_prob <- emp_trans / rowSums(emp_trans)
cat("\nEmpirical transition probs:\n")
print(emp_trans_prob)

# Save statistics
rows <- list()
idx <- 1

rows[[idx]] <- data.frame(key = "n_sequences", value = n_seq); idx <- idx + 1
rows[[idx]] <- data.frame(key = "sequence_length", value = seq_len); idx <- idx + 1
rows[[idx]] <- data.frame(key = "n_states", value = n_states); idx <- idx + 1
rows[[idx]] <- data.frame(key = "n_symbols", value = ncol(emiss_sim)); idx <- idx + 1

# Symbol frequencies
for (sym in names(symbol_freq)) {
  rows[[idx]] <- data.frame(key = paste0("symbol_freq_", sym),
                            value = as.numeric(symbol_freq[sym]))
  idx <- idx + 1
}

# State frequencies
for (st in names(state_freq)) {
  rows[[idx]] <- data.frame(key = paste0("state_freq_", st),
                            value = as.numeric(state_freq[st]))
  idx <- idx + 1
}

# Empirical transition probs
for (i in 1:n_states) {
  for (j in 1:n_states) {
    rows[[idx]] <- data.frame(
      key = paste0("emp_trans_", i - 1, "_", j - 1),
      value = emp_trans_prob[i, j]
    )
    idx <- idx + 1
  }
}

out_stats <- do.call(rbind, rows)
write.csv(out_stats, file.path(outdir, "ref_sim_hmm_stats.csv"),
          row.names = FALSE)
cat("Written: ref_sim_hmm_stats.csv\n")

# ======================================================================
# MHMM Simulation: 2 clusters
# ======================================================================
cat("\n=== MHMM Simulation ===\n")

init_1 <- c(0.8, 0.2)
trans_1 <- matrix(c(0.9, 0.1, 0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_1 <- matrix(c(0.7, 0.2, 0.1,
                     0.1, 0.3, 0.6), nrow = 2, byrow = TRUE)
colnames(emiss_1) <- c("A", "B", "C")

init_2 <- c(0.3, 0.7)
trans_2 <- matrix(c(0.6, 0.4, 0.1, 0.9), nrow = 2, byrow = TRUE)
emiss_2 <- matrix(c(0.1, 0.2, 0.7,
                     0.6, 0.3, 0.1), nrow = 2, byrow = TRUE)
colnames(emiss_2) <- c("A", "B", "C")

n_mhmm <- 50

set.seed(123)
sim_mhmm <- simulate_mhmm(
  n_sequences = n_mhmm,
  initial_probs = list(init_1, init_2),
  transition_probs = list(trans_1, trans_2),
  emission_probs = list(emiss_1, emiss_2),
  sequence_length = seq_len
)

cat("class(sim_mhmm):", class(sim_mhmm), "\n")
cat("names(sim_mhmm):", paste(names(sim_mhmm), collapse = ", "), "\n")

# Extract and compute stats for MHMM
mhmm_obs <- sim_mhmm$observations
mhmm_obs_mat <- as.matrix(mhmm_obs)
mhmm_obs_vec <- as.vector(mhmm_obs_mat)
mhmm_sym_freq <- table(mhmm_obs_vec) / length(mhmm_obs_vec)
cat("\nMHMM Symbol frequencies:\n")
print(mhmm_sym_freq)

rows_m <- list()
idx <- 1
rows_m[[idx]] <- data.frame(key = "n_sequences", value = n_mhmm); idx <- idx + 1
rows_m[[idx]] <- data.frame(key = "sequence_length", value = seq_len); idx <- idx + 1
rows_m[[idx]] <- data.frame(key = "n_clusters", value = 2); idx <- idx + 1

for (sym in names(mhmm_sym_freq)) {
  rows_m[[idx]] <- data.frame(key = paste0("symbol_freq_", sym),
                              value = as.numeric(mhmm_sym_freq[sym]))
  idx <- idx + 1
}

out_mhmm <- do.call(rbind, rows_m)
write.csv(out_mhmm, file.path(outdir, "ref_sim_mhmm_stats.csv"),
          row.names = FALSE)
cat("Written: ref_sim_mhmm_stats.csv\n")

cat("\nAll simulation reference files written to:", outdir, "\n")
