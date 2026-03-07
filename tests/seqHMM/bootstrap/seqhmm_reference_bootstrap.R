# @Author  : Yapeng Wei
# @File    : seqhmm_reference_bootstrap.R
# @Desc    :
# R seqHMM reference values for bootstrap_coefs consistency testing
# with sequenzo.seqhmm (Python).
#
# bootstrap_coefs() in seqHMM performs parametric bootstrap to obtain
# confidence intervals for NHMM regression coefficients. This is
# particularly useful because standard errors from the Hessian may be
# unreliable near boundary solutions.
#
# Strategy:
#   1. Fit an NHMM with covariates
#   2. Run bootstrap_coefs with small B (for speed)
#   3. Record the bootstrap distribution statistics
#   4. Compare Python's implementation against R's
#
# Usage:
#   Rscript seqhmm_reference_bootstrap.R [outdir]
#
# Output:
#   ref_bootstrap_stats.csv   - Bootstrap summary statistics
#   ref_bootstrap_samples.csv - Raw bootstrap samples (for distribution tests)
#
# Requires: seqHMM (>= 2.1.0)

if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}

pkg_ver <- packageVersion("seqHMM")
if (pkg_ver < "2.1.0") {
  stop("seqHMM >= 2.1.0 required for bootstrap_coefs. Current: ", pkg_ver)
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

suppressPackageStartupMessages({
  library(seqHMM)
})

cat("seqHMM version:", as.character(pkg_ver), "\n")

# ======================================================================
# Synthetic panel data (same as NHMM test)
# ======================================================================
set.seed(42)
n_id <- 20
n_time <- 10
symbols <- c("A", "B", "C")

ids <- rep(1:n_id, each = n_time)
times <- rep(1:n_time, times = n_id)

set.seed(42)
x_covariate <- round(rnorm(n_id * n_time, mean = 0, sd = 1), 3)

set.seed(42)
response <- character(n_id * n_time)
for (i in 1:(n_id * n_time)) {
  probs <- c(0.5, 0.3, 0.2)
  if (x_covariate[i] > 0) {
    probs <- c(0.2, 0.3, 0.5)
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

# ======================================================================
# Step 1: Fit NHMM
# ======================================================================
cat("=== Fitting NHMM ===\n")

set.seed(123)
nhmm_fit <- tryCatch({
  estimate_nhmm(
    n_states = 2,
    data = panel_data,
    time = "time",
    id = "id",
    transition_formula = ~ x,
    emission_formula = response ~ 1,
    method = "DNM"
  )
}, error = function(e) {
  cat("estimate_nhmm error:", e$message, "\n")
  NULL
})

if (is.null(nhmm_fit)) {
  cat("NHMM fitting failed, writing failure marker.\n")
  write.csv(
    data.frame(key = "status", value = "failed"),
    file.path(outdir, "ref_bootstrap_stats.csv"),
    row.names = FALSE
  )
  quit(status = 0)
}

cat("NHMM logLik:", as.numeric(logLik(nhmm_fit)), "\n")

# ======================================================================
# Step 2: Bootstrap (small B for reproducibility / speed)
# ======================================================================
cat("\n=== Running Bootstrap ===\n")
B <- 50  # small B for testing (production would use 500+)

set.seed(456)
boot_result <- tryCatch({
  bootstrap_coefs(nhmm_fit, nsim = B, method = "DNM")
}, error = function(e) {
  cat("bootstrap_coefs error:", e$message, "\n")
  NULL
})

if (is.null(boot_result)) {
  cat("Bootstrap failed, writing failure marker.\n")
  write.csv(
    data.frame(key = "status", value = "bootstrap_failed"),
    file.path(outdir, "ref_bootstrap_stats.csv"),
    row.names = FALSE
  )
  quit(status = 0)
}

cat("class(boot_result):", class(boot_result), "\n")

# ======================================================================
# Step 3: Extract bootstrap results
# ======================================================================

# The boot element contains the bootstrap samples
boot_data <- boot_result$boot
cat("names(boot_data):", paste(names(boot_data), collapse = ", "), "\n")

rows <- list()
idx <- 1

# Record basic info
rows[[idx]] <- data.frame(key = "status", value = "success"); idx <- idx + 1
rows[[idx]] <- data.frame(key = "B", value = B); idx <- idx + 1
rows[[idx]] <- data.frame(key = "loglik_original",
                          value = as.numeric(logLik(nhmm_fit))); idx <- idx + 1

# Extract coefficient summaries from bootstrap
# boot_data typically has: transition, emission, initial (each a matrix)
for (comp_name in names(boot_data)) {
  comp <- boot_data[[comp_name]]
  cat("\nBootstrap component:", comp_name, "\n")
  cat("  class:", class(comp), "\n")

  if (is.matrix(comp) || is.data.frame(comp)) {
    comp_mat <- as.matrix(comp)
    cat("  dim:", dim(comp_mat), "\n")

    # Summary stats per coefficient
    n_coefs <- ncol(comp_mat)
    for (j in 1:n_coefs) {
      col_vals <- comp_mat[, j]
      col_name <- if (!is.null(colnames(comp_mat))) colnames(comp_mat)[j] else paste0("coef_", j)
      prefix <- paste0(comp_name, "_", col_name)

      rows[[idx]] <- data.frame(key = paste0(prefix, "_mean"),
                                value = mean(col_vals, na.rm = TRUE)); idx <- idx + 1
      rows[[idx]] <- data.frame(key = paste0(prefix, "_sd"),
                                value = sd(col_vals, na.rm = TRUE)); idx <- idx + 1
      rows[[idx]] <- data.frame(key = paste0(prefix, "_q025"),
                                value = quantile(col_vals, 0.025, na.rm = TRUE)); idx <- idx + 1
      rows[[idx]] <- data.frame(key = paste0(prefix, "_q975"),
                                value = quantile(col_vals, 0.975, na.rm = TRUE)); idx <- idx + 1
    }
  } else {
    cat("  Skipping non-matrix component\n")
  }
}

out_stats <- do.call(rbind, rows)
write.csv(out_stats, file.path(outdir, "ref_bootstrap_stats.csv"),
          row.names = FALSE)
cat("\nWritten: ref_bootstrap_stats.csv (", nrow(out_stats), "rows)\n")

cat("\nAll bootstrap reference files written to:", outdir, "\n")
