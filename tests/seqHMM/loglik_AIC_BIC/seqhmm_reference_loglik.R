# @Author  : Yapeng Wei
# @File    : seqhmm_reference_loglik.R
# @Desc    :
# R seqHMM reference values for logLik, AIC, BIC consistency testing with
# sequenzo.seqhmm (Python).
#
# Generates reference values from R seqHMM using synthetic data with
# manually specified HMM parameters (no randomness, fully deterministic).
#
# Usage:
#   Rscript seqhmm_reference_loglik.R [outdir]
#
# Output:
#   ref_loglik.csv         - logLik, df, nobs, AIC, BIC for each config
#   ref_hidden_paths_A.csv - Viterbi paths for Config A (for future tests)
#   ref_posterior_A.csv    - posterior probs for Config A (for future tests)
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
cat("TraMineR version:", as.character(packageVersion("TraMineR")), "\n")

# ======================================================================
# Synthetic test data
# ======================================================================
# 5 sequences, 8 time points, 3 observed states (A, B, C).
# This data is shared with the Python test (must be IDENTICAL).
#
# Seq s0: A A B B C A A B
# Seq s1: B C C A A B C C
# Seq s2: A B C A B C A B
# Seq s3: C C B A A A B C
# Seq s4: A A A B B B C C

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

# Alphabet must be explicitly set to A, B, C in this order.
# This order determines emission matrix column mapping:
#   column 1 = A, column 2 = B, column 3 = C
seq_obj <- seqdef(seq_data, alphabet = c("A", "B", "C"),
                  id = paste0("s", 0:4))

# ======================================================================
# Config A: 2 hidden states, basic parameters
# ======================================================================
# Emission matrix interpretation:
#   emiss[1,1]=0.5  P(emit "A" | State 1) = 0.5
#   emiss[1,2]=0.3  P(emit "B" | State 1) = 0.3
#   emiss[1,3]=0.2  P(emit "C" | State 1) = 0.2
#   emiss[2,1]=0.1  P(emit "A" | State 2) = 0.1
#   emiss[2,2]=0.4  P(emit "B" | State 2) = 0.4
#   emiss[2,3]=0.5  P(emit "C" | State 2) = 0.5

init_A <- c(0.6, 0.4)
trans_A <- matrix(c(
  0.7, 0.3,
  0.2, 0.8
), nrow = 2, byrow = TRUE)
emiss_A <- matrix(c(
  0.5, 0.3, 0.2,
  0.1, 0.4, 0.5
), nrow = 2, byrow = TRUE)

hmm_A <- build_hmm(observations = seq_obj,
                    transition_probs = trans_A,
                    emission_probs = emiss_A,
                    initial_probs = init_A)

ll_A_obj <- logLik(hmm_A)
ll_A     <- as.numeric(ll_A_obj)
df_A     <- attr(ll_A_obj, "df")
nobs_A   <- attr(ll_A_obj, "nobs")
aic_A    <- AIC(hmm_A)
bic_A    <- BIC(hmm_A)

cat("\n--- Config A (2 states, basic) ---\n")
cat("logLik:", ll_A, "\n")
cat("df:", df_A, "  nobs:", nobs_A, "\n")
cat("AIC:", aic_A, "  BIC:", bic_A, "\n")

# ======================================================================
# Config B: 3 hidden states
# ======================================================================
init_B <- c(0.5, 0.3, 0.2)
trans_B <- matrix(c(
  0.7, 0.2, 0.1,
  0.1, 0.7, 0.2,
  0.2, 0.1, 0.7
), nrow = 3, byrow = TRUE)
emiss_B <- matrix(c(
  0.6, 0.3, 0.1,
  0.1, 0.6, 0.3,
  0.3, 0.1, 0.6
), nrow = 3, byrow = TRUE)

hmm_B <- build_hmm(observations = seq_obj,
                    transition_probs = trans_B,
                    emission_probs = emiss_B,
                    initial_probs = init_B)

ll_B_obj <- logLik(hmm_B)
ll_B     <- as.numeric(ll_B_obj)
df_B     <- attr(ll_B_obj, "df")
nobs_B   <- attr(ll_B_obj, "nobs")
aic_B    <- AIC(hmm_B)
bic_B    <- BIC(hmm_B)

cat("\n--- Config B (3 states) ---\n")
cat("logLik:", ll_B, "\n")
cat("df:", df_B, "  nobs:", nobs_B, "\n")
cat("AIC:", aic_B, "  BIC:", bic_B, "\n")

# ======================================================================
# Config C: 2 hidden states, high self-transition (sticky model)
# ======================================================================
init_C <- c(0.9, 0.1)
trans_C <- matrix(c(
  0.95, 0.05,
  0.05, 0.95
), nrow = 2, byrow = TRUE)
emiss_C <- matrix(c(
  0.8, 0.15, 0.05,
  0.05, 0.15, 0.8
), nrow = 2, byrow = TRUE)

hmm_C <- build_hmm(observations = seq_obj,
                    transition_probs = trans_C,
                    emission_probs = emiss_C,
                    initial_probs = init_C)

ll_C_obj <- logLik(hmm_C)
ll_C     <- as.numeric(ll_C_obj)
df_C     <- attr(ll_C_obj, "df")
nobs_C   <- attr(ll_C_obj, "nobs")
aic_C    <- AIC(hmm_C)
bic_C    <- BIC(hmm_C)

cat("\n--- Config C (2 states, sticky) ---\n")
cat("logLik:", ll_C, "\n")
cat("df:", df_C, "  nobs:", nobs_C, "\n")
cat("AIC:", aic_C, "  BIC:", bic_C, "\n")

# ======================================================================
# Config D: 2 hidden states, uniform emissions (null model baseline)
# ======================================================================
# With uniform emissions, the HMM cannot distinguish states from
# observations. logLik should equal n_obs * log(1/n_symbols) = 40*log(1/3).
init_D <- c(0.5, 0.5)
trans_D <- matrix(c(
  0.5, 0.5,
  0.5, 0.5
), nrow = 2, byrow = TRUE)
emiss_D <- matrix(c(
  1/3, 1/3, 1/3,
  1/3, 1/3, 1/3
), nrow = 2, byrow = TRUE)

hmm_D <- build_hmm(observations = seq_obj,
                    transition_probs = trans_D,
                    emission_probs = emiss_D,
                    initial_probs = init_D)

ll_D_obj <- logLik(hmm_D)
ll_D     <- as.numeric(ll_D_obj)
df_D     <- attr(ll_D_obj, "df")
nobs_D   <- attr(ll_D_obj, "nobs")
aic_D    <- AIC(hmm_D)
bic_D    <- BIC(hmm_D)

cat("\n--- Config D (uniform, null baseline) ---\n")
cat("logLik:", ll_D, "\n")
cat("df:", df_D, "  nobs:", nobs_D, "\n")
cat("AIC:", aic_D, "  BIC:", bic_D, "\n")

# ======================================================================
# Write all results to CSV
# ======================================================================
results <- data.frame(
  config   = c("A", "B", "C", "D"),
  n_states = c(2L, 3L, 2L, 2L),
  loglik   = c(ll_A, ll_B, ll_C, ll_D),
  df       = c(df_A, df_B, df_C, df_D),
  nobs     = c(nobs_A, nobs_B, nobs_C, nobs_D),
  aic      = c(aic_A, aic_B, aic_C, aic_D),
  bic      = c(bic_A, bic_B, bic_C, bic_D)
)

write.csv(results, file.path(outdir, "ref_loglik.csv"), row.names = FALSE)

# ======================================================================
# (Future use) Hidden paths for Config A (Viterbi)
# ======================================================================
hp_A <- hidden_paths(hmm_A)
write.csv(hp_A, file.path(outdir, "ref_hidden_paths_A.csv"), row.names = FALSE)

# ======================================================================
# (Future use) Posterior probabilities for Config A (forward-backward)
# ======================================================================
post_A <- posterior_probs(hmm_A)
write.csv(post_A, file.path(outdir, "ref_posterior_A.csv"), row.names = FALSE)

cat("\nAll reference files written to:", outdir, "\n")
cat("Files: ref_loglik.csv, ref_hidden_paths_A.csv, ref_posterior_A.csv\n")
