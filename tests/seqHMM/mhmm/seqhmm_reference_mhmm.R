# @Author  : Yapeng Wei
# @File    : seqhmm_reference_mhmm.R
# @Desc    :
# R seqHMM reference values for Mixture Hidden Markov Model (MHMM)
# consistency testing with sequenzo.seqhmm (Python).
#
# Tests build_mhmm, logLik, hidden_paths, posterior_probs,
# cluster_probs, separate_mhmm, and fit_model for MHMM objects.
#
# Data: 10 sequences x 8 timepoints x 3 observed states (A,B,C),
# 2 clusters with different HMM parameters.
#
# Usage:
#   Rscript seqhmm_reference_mhmm.R [outdir]
#
# Output:
#   ref_mhmm_loglik.csv       - logLik values for MHMM configs
#   ref_mhmm_hidden_paths.csv - Viterbi paths per sequence
#   ref_mhmm_cluster.csv      - cluster probabilities & assignments
#   ref_mhmm_em.csv           - EM converged logLik
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
# Synthetic test data: 10 sequences, 8 timepoints, alphabet {A,B,C}
# First 5 are same as basic HMM tests; 5 more added for MHMM
# ======================================================================
seq_data <- data.frame(
  t1 = c("A", "B", "A", "C", "A", "C", "C", "B", "A", "C"),
  t2 = c("A", "C", "B", "C", "A", "C", "B", "B", "A", "B"),
  t3 = c("B", "C", "C", "B", "A", "B", "C", "C", "B", "A"),
  t4 = c("B", "A", "A", "A", "B", "A", "A", "C", "B", "A"),
  t5 = c("C", "A", "B", "A", "B", "A", "B", "B", "C", "B"),
  t6 = c("A", "B", "C", "A", "B", "C", "C", "A", "A", "C"),
  t7 = c("A", "C", "A", "B", "C", "B", "A", "B", "A", "A"),
  t8 = c("B", "C", "B", "C", "C", "C", "B", "C", "B", "B"),
  stringsAsFactors = FALSE
)
seq_obj <- seqdef(seq_data, alphabet = c("A", "B", "C"),
                  id = paste0("s", 0:9))

# ======================================================================
# MHMM Config: 2 clusters, each with 2 hidden states
# Cluster 1: prefers A emissions in state 1
# Cluster 2: prefers C emissions in state 1
# ======================================================================

# Cluster 1 parameters
init_1 <- c(0.7, 0.3)
trans_1 <- matrix(c(0.8, 0.2, 0.3, 0.7), nrow = 2, byrow = TRUE)
emiss_1 <- matrix(c(0.6, 0.3, 0.1,
                     0.1, 0.4, 0.5), nrow = 2, byrow = TRUE)

# Cluster 2 parameters
init_2 <- c(0.4, 0.6)
trans_2 <- matrix(c(0.6, 0.4, 0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_2 <- matrix(c(0.1, 0.3, 0.6,
                     0.5, 0.4, 0.1), nrow = 2, byrow = TRUE)

# Build MHMM (no covariates)
mhmm <- build_mhmm(
  observations = seq_obj,
  initial_probs = list(init_1, init_2),
  transition_probs = list(trans_1, trans_2),
  emission_probs = list(emiss_1, emiss_2),
  cluster_names = c("Cluster1", "Cluster2")
)

cat("\nclass(mhmm):", class(mhmm), "\n")

# ======================================================================
# 1. Log-likelihood
# ======================================================================
ll <- as.numeric(logLik(mhmm))
cat("MHMM logLik:", ll, "\n")

rows <- list()
idx <- 1
rows[[idx]] <- data.frame(key = "loglik", value = ll); idx <- idx + 1

# Number of free parameters (df)
n_params <- attr(logLik(mhmm), "df")
cat("MHMM n_params (df):", n_params, "\n")
rows[[idx]] <- data.frame(key = "n_params", value = n_params); idx <- idx + 1

# AIC and BIC
aic_val <- AIC(mhmm)
bic_val <- BIC(mhmm)
cat("MHMM AIC:", aic_val, "\n")
cat("MHMM BIC:", bic_val, "\n")
rows[[idx]] <- data.frame(key = "aic", value = aic_val); idx <- idx + 1
rows[[idx]] <- data.frame(key = "bic", value = bic_val); idx <- idx + 1

# nobs
n_obs <- nobs(mhmm)
cat("MHMM nobs:", n_obs, "\n")
rows[[idx]] <- data.frame(key = "nobs", value = n_obs); idx <- idx + 1

out_loglik <- do.call(rbind, rows)
write.csv(out_loglik, file.path(outdir, "ref_mhmm_loglik.csv"),
          row.names = FALSE)
cat("Written: ref_mhmm_loglik.csv\n")

# ======================================================================
# 2. Hidden paths (Viterbi)
# ======================================================================
hp <- hidden_paths(mhmm)
cat("\nhidden_paths result:\n")
print(hp)

# Extract state columns (data.table format in newer versions)
if (is.data.frame(hp)) {
  # seqHMM 2.x returns data.table (inherits data.frame) with id, time, state columns
  hp_df <- as.data.frame(hp)
  cat("  hidden_paths columns:", paste(names(hp_df), collapse=", "), "\n")
  write.csv(hp_df, file.path(outdir, "ref_mhmm_hidden_paths.csv"),
            row.names = FALSE)
} else {
  # Older format: stslist
  hp_mat <- as.matrix(hp)
  hp_df <- data.frame(
    id = rownames(hp_mat),
    hp_mat,
    stringsAsFactors = FALSE
  )
  write.csv(hp_df, file.path(outdir, "ref_mhmm_hidden_paths.csv"),
            row.names = FALSE)
}
cat("Written: ref_mhmm_hidden_paths.csv\n")

# ======================================================================
# 3. Cluster probabilities & most probable cluster
# ======================================================================
# posterior_cluster_probabilities or most_probable_cluster
rows_cl <- list()
idx <- 1

# Try to get cluster assignment info
tryCatch({
  # In seqHMM 2.x, summary provides cluster info
  s <- summary(mhmm)
  cat("\nsummary(mhmm):\n")
  print(s)
}, error = function(e) {
  cat("summary error:", e$message, "\n")
})

# Try posterior_cluster_probabilities
tryCatch({
  pcp <- posterior_cluster_probabilities(mhmm)
  cat("\nposterior_cluster_probabilities:\n")
  print(pcp)
  pcp_df <- as.data.frame(pcp)
  write.csv(pcp_df, file.path(outdir, "ref_mhmm_cluster.csv"),
            row.names = FALSE)
  cat("Written: ref_mhmm_cluster.csv\n")
}, error = function(e) {
  cat("posterior_cluster_probabilities error:", e$message, "\n")
  # Fallback: try most_probable_cluster
  tryCatch({
    mpc <- most_probable_cluster(mhmm)
    cat("\nmost_probable_cluster:\n")
    print(mpc)
    mpc_df <- data.frame(cluster = as.character(mpc))
    write.csv(mpc_df, file.path(outdir, "ref_mhmm_cluster.csv"),
              row.names = FALSE)
    cat("Written: ref_mhmm_cluster.csv (from most_probable_cluster)\n")
  }, error = function(e2) {
    cat("most_probable_cluster error:", e2$message, "\n")
  })
})

# ======================================================================
# 4. EM fitting
# ======================================================================
cat("\n=== MHMM EM Fitting ===\n")
ll_before <- as.numeric(logLik(mhmm))
cat("logLik before EM:", ll_before, "\n")

fitted <- fit_model(
  mhmm,
  em_step = TRUE,
  global_step = FALSE,
  local_step = FALSE,
  control_em = list(
    maxeval = 1000,
    reltol = 1e-10,
    print_level = 0
  )
)

m <- fitted$model
ll_after <- as.numeric(logLik(m))
cat("logLik after EM:", ll_after, "\n")

em_df <- data.frame(
  key = c("loglik_before", "loglik_after"),
  value = c(ll_before, ll_after)
)
write.csv(em_df, file.path(outdir, "ref_mhmm_em.csv"), row.names = FALSE)
cat("Written: ref_mhmm_em.csv\n")

# ======================================================================
# 5. Separate MHMM into individual HMMs
# ======================================================================
tryCatch({
  sep <- separate_mhmm(mhmm)
  cat("\nseparate_mhmm: class =", class(sep), ", length =", length(sep), "\n")
  for (i in seq_along(sep)) {
    cat("  Cluster", i, ": logLik =", as.numeric(logLik(sep[[i]])), "\n")
  }
}, error = function(e) {
  cat("separate_mhmm error:", e$message, "\n")
})

cat("\nAll MHMM reference files written to:", outdir, "\n")
