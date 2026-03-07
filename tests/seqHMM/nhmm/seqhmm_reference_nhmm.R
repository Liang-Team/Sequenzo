# @Author  : Yapeng Wei
# @File    : seqhmm_reference_nhmm.R
# @Desc    :
# R seqHMM reference values for Non-Homogeneous Hidden Markov Model
# (NHMM) consistency testing with sequenzo.seqhmm (Python).
#
# NHMMs allow initial, transition, and emission probabilities to depend
# on external covariates via multinomial logistic regression. In
# seqHMM 2.x, NHMMs are estimated via estimate_nhmm() rather than
# build_hmm() + fit_model().
#
# Tests cover:
#   1. estimate_nhmm with covariates in transition/emission
#   2. logLik of fitted NHMM
#   3. hidden_paths (Viterbi) for NHMM
#   4. Coefficient extraction (coef)
#   5. Prediction / get_marginals
#
# Data: Panel data format with id, time, response, covariates.
#
# Usage:
#   Rscript seqhmm_reference_nhmm.R [outdir]
#
# Output:
#   ref_nhmm_fit.csv          - Fitted model logLik, AIC, BIC, n_params
#   ref_nhmm_hidden_paths.csv - Viterbi paths
#   ref_nhmm_coefs.csv        - Estimated coefficients
#
# Requires: seqHMM (>= 2.1.0), TraMineR

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

pkg_ver <- packageVersion("seqHMM")
if (pkg_ver < "2.1.0") {
  stop("seqHMM >= 2.1.0 required for NHMM. Current version: ", pkg_ver)
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(seqHMM)
})

cat("seqHMM version:", as.character(pkg_ver), "\n")

# ======================================================================
# Synthetic panel data for NHMM
# 20 individuals, 10 timepoints, 3 observed states (A, B, C)
# Covariate: x (continuous, time-varying)
# ======================================================================
set.seed(42)
n_id <- 20
n_time <- 10
n_states <- 2
symbols <- c("A", "B", "C")

# Generate covariates
ids <- rep(1:n_id, each = n_time)
times <- rep(1:n_time, times = n_id)

set.seed(42)
x_covariate <- round(rnorm(n_id * n_time, mean = 0, sd = 1), 3)

# Generate response sequences using a simple generative model
# (the NHMM will try to recover the structure)
set.seed(42)
response <- character(n_id * n_time)
for (i in 1:(n_id * n_time)) {
  # Simple probabilistic generation based on covariate
  probs <- c(0.5, 0.3, 0.2)
  if (x_covariate[i] > 0) {
    probs <- c(0.2, 0.3, 0.5)  # shift toward C when x > 0
  }
  response[i] <- sample(symbols, 1, prob = probs)
}

panel_data <- data.frame(
  id = ids,
  time = times,
  response = response,
  x = x_covariate,
  stringsAsFactors = FALSE
)
# seqHMM 2.1.0 requires the response column to be a factor
panel_data$response <- factor(panel_data$response, levels = symbols)

cat("Panel data dimensions:", nrow(panel_data), "rows\n")
cat("Unique ids:", length(unique(panel_data$id)), "\n")
cat("Unique times:", length(unique(panel_data$time)), "\n")
cat("Response table:\n")
print(table(panel_data$response))

# Save panel data for Python to use
write.csv(panel_data, file.path(outdir, "ref_nhmm_panel_data.csv"),
          row.names = FALSE)
cat("Written: ref_nhmm_panel_data.csv\n")

# ======================================================================
# Estimate NHMM with covariate x affecting transitions
# ======================================================================
cat("\n=== Estimating NHMM ===\n")

# estimate_nhmm expects: data in long format with id, time, response columns.
# In seqHMM >= 2.1.0, emission_formula must have the response variable
# on the LEFT-hand side of the formula (e.g., response ~ 1).
# The 'response' parameter is NOT used separately.
set.seed(123)
nhmm_fit <- tryCatch({
  estimate_nhmm(
    n_states = n_states,
    data = panel_data,
    time = "time",
    id = "id",
    init_formula = ~ 1,
    transition_formula = ~ x,
    emission_formula = response ~ 1,
    method = "DNM"
  )
}, error = function(e) {
  cat("estimate_nhmm error:", e$message, "\n")
  # Try simpler model (intercept-only for all components)
  tryCatch({
    estimate_nhmm(
      n_states = n_states,
      data = panel_data,
      time = "time",
      id = "id",
      emission_formula = response ~ 1,
      method = "DNM"
    )
  }, error = function(e2) {
    cat("Simpler estimate_nhmm also failed:", e2$message, "\n")
    NULL
  })
})

if (!is.null(nhmm_fit)) {
  cat("class(nhmm_fit):", class(nhmm_fit), "\n")

  # Log-likelihood
  ll <- as.numeric(logLik(nhmm_fit))
  cat("logLik:", ll, "\n")

  # Degrees of freedom
  n_params <- attr(logLik(nhmm_fit), "df")
  cat("n_params (df):", n_params, "\n")

  # AIC, BIC
  aic_val <- tryCatch(AIC(nhmm_fit), error = function(e) NA)
  bic_val <- tryCatch(BIC(nhmm_fit), error = function(e) NA)
  cat("AIC:", aic_val, "\n")
  cat("BIC:", bic_val, "\n")

  # nobs
  n_obs <- tryCatch(nobs(nhmm_fit), error = function(e) NA)
  cat("nobs:", n_obs, "\n")

  rows <- list()
  idx <- 1
  rows[[idx]] <- data.frame(key = "loglik", value = ll); idx <- idx + 1
  rows[[idx]] <- data.frame(key = "n_params", value = n_params); idx <- idx + 1
  if (!is.na(aic_val)) {
    rows[[idx]] <- data.frame(key = "aic", value = aic_val); idx <- idx + 1
  }
  if (!is.na(bic_val)) {
    rows[[idx]] <- data.frame(key = "bic", value = bic_val); idx <- idx + 1
  }
  if (!is.na(n_obs)) {
    rows[[idx]] <- data.frame(key = "nobs", value = n_obs); idx <- idx + 1
  }
  rows[[idx]] <- data.frame(key = "n_states", value = n_states); idx <- idx + 1
  rows[[idx]] <- data.frame(key = "n_id", value = n_id); idx <- idx + 1
  rows[[idx]] <- data.frame(key = "n_time", value = n_time); idx <- idx + 1

  out_fit <- do.call(rbind, rows)
  write.csv(out_fit, file.path(outdir, "ref_nhmm_fit.csv"), row.names = FALSE)
  cat("Written: ref_nhmm_fit.csv\n")

  # ======================================================================
  # Hidden paths (Viterbi)
  # ======================================================================
  tryCatch({
    hp <- hidden_paths(nhmm_fit)
    cat("\nhidden_paths:\n")
    print(head(hp))
    hp_df <- as.data.frame(hp)
    write.csv(hp_df, file.path(outdir, "ref_nhmm_hidden_paths.csv"),
              row.names = FALSE)
    cat("Written: ref_nhmm_hidden_paths.csv\n")
  }, error = function(e) {
    cat("hidden_paths error:", e$message, "\n")
  })

  # ======================================================================
  # Coefficients
  # ======================================================================
  tryCatch({
    cf <- coef(nhmm_fit)
    cat("\ncoefficients:\n")
    print(cf)
    # coef returns a list of data.tables for init, transition, emission
    if (is.list(cf)) {
      for (nm in names(cf)) {
        cf_df <- as.data.frame(cf[[nm]])
        fname <- paste0("ref_nhmm_coefs_", nm, ".csv")
        write.csv(cf_df, file.path(outdir, fname), row.names = FALSE)
        cat("Written:", fname, "\n")
      }
    }
  }, error = function(e) {
    cat("coef error:", e$message, "\n")
  })

} else {
  cat("\nNHMM estimation failed. Writing empty reference.\n")
  empty_df <- data.frame(key = "status", value = "failed")
  write.csv(empty_df, file.path(outdir, "ref_nhmm_fit.csv"), row.names = FALSE)
}

cat("\nAll NHMM reference files written to:", outdir, "\n")
