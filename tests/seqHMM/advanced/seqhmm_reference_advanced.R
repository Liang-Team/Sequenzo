# @Author  : Yapeng Wei
# @File    : seqhmm_reference_advanced.R
# @Desc    :
# R seqHMM reference values for advanced features consistency testing
# with sequenzo.seqhmm (Python).
#
# Tests cover:
#   1. fit_model with em_step + global_step (EM then DNM)
#   2. fit_model with em_step + local_step (EM then local opt)
#   3. fit_model with restart (multiple random starts)
#   4. Formula-based MHMM with covariates
#   5. trim_model (zero out small probabilities)
#   6. BIC-based model comparison across different n_states
#
# Uses the same 5-sequence test data as loglik/posterior/viterbi tests.
#
# Usage:
#   Rscript seqhmm_reference_advanced.R [outdir]
#
# Output:
#   ref_advanced_fit.csv          - fit_model variant results
#   ref_advanced_formula.csv      - Formula-based MHMM results
#   ref_advanced_trim.csv         - trim_model results
#   ref_advanced_comparison.csv   - Model comparison (BIC)
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
# Synthetic test data (same as basic HMM tests)
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

# Covariate for Formula-based MHMM (10 sequences)
seq_data_10 <- data.frame(
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
seq_obj_10 <- seqdef(seq_data_10, alphabet = c("A", "B", "C"),
                     id = paste0("s", 0:9))

# Covariates for Formula MHMM
covariate_data <- data.frame(
  x = c(1.2, -0.5, 0.3, 1.8, -1.1, 0.7, -0.3, 0.9, -0.8, 1.5)
)

# ======================================================================
# Config A (2 states) for advanced fitting tests
# ======================================================================
init_A <- c(0.6, 0.4)
trans_A <- matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_A <- matrix(c(0.5, 0.3, 0.2, 0.1, 0.4, 0.5), nrow = 2, byrow = TRUE)

rows <- list()
idx <- 1

# ======================================================================
# 1. fit_model: EM only
# ======================================================================
cat("\n=== EM only ===\n")
hmm_em <- build_hmm(observations = seq_obj,
                     transition_probs = trans_A,
                     emission_probs = emiss_A,
                     initial_probs = init_A)
fit_em <- fit_model(hmm_em,
                    em_step = TRUE,
                    global_step = FALSE,
                    local_step = FALSE,
                    control_em = list(maxeval = 1000, reltol = 1e-10,
                                     print_level = 0))
ll_em <- as.numeric(logLik(fit_em$model))
cat("  EM logLik:", ll_em, "\n")
rows[[idx]] <- data.frame(key = "loglik_em_only", value = ll_em); idx <- idx + 1

# ======================================================================
# 2. fit_model: EM + global step
# ======================================================================
cat("\n=== EM + Global step ===\n")
hmm_eg <- build_hmm(observations = seq_obj,
                     transition_probs = trans_A,
                     emission_probs = emiss_A,
                     initial_probs = init_A)
fit_eg <- tryCatch({
  fit_model(hmm_eg,
            em_step = TRUE,
            global_step = TRUE,
            local_step = FALSE,
            control_em = list(maxeval = 500, reltol = 1e-10,
                              print_level = 0),
            control_global = list(maxeval = 500))
}, error = function(e) {
  cat("  EM+Global error:", e$message, "\n")
  NULL
})

if (!is.null(fit_eg)) {
  ll_eg <- as.numeric(logLik(fit_eg$model))
  cat("  EM+Global logLik:", ll_eg, "\n")
  rows[[idx]] <- data.frame(key = "loglik_em_global", value = ll_eg); idx <- idx + 1
} else {
  rows[[idx]] <- data.frame(key = "loglik_em_global", value = NA); idx <- idx + 1
}

# ======================================================================
# 3. fit_model: EM + local step
# ======================================================================
cat("\n=== EM + Local step ===\n")
hmm_el <- build_hmm(observations = seq_obj,
                     transition_probs = trans_A,
                     emission_probs = emiss_A,
                     initial_probs = init_A)
fit_el <- tryCatch({
  fit_model(hmm_el,
            em_step = TRUE,
            global_step = FALSE,
            local_step = TRUE,
            control_em = list(maxeval = 500, reltol = 1e-10,
                              print_level = 0),
            control_local = list(maxeval = 500))
}, error = function(e) {
  cat("  EM+Local error:", e$message, "\n")
  NULL
})

if (!is.null(fit_el)) {
  ll_el <- as.numeric(logLik(fit_el$model))
  cat("  EM+Local logLik:", ll_el, "\n")
  rows[[idx]] <- data.frame(key = "loglik_em_local", value = ll_el); idx <- idx + 1
} else {
  rows[[idx]] <- data.frame(key = "loglik_em_local", value = NA); idx <- idx + 1
}

# ======================================================================
# 4. fit_model with restart (random starts)
# ======================================================================
cat("\n=== EM with restart ===\n")
hmm_restart <- build_hmm(observations = seq_obj,
                          transition_probs = trans_A,
                          emission_probs = emiss_A,
                          initial_probs = init_A)
set.seed(42)
fit_restart <- tryCatch({
  fit_model(hmm_restart,
            em_step = TRUE,
            global_step = FALSE,
            local_step = FALSE,
            control_em = list(
              maxeval = 500,
              reltol = 1e-10,
              print_level = 0,
              restart = list(times = 5, transition = TRUE,
                             emission = TRUE, sd = 0.25)
            ))
}, error = function(e) {
  cat("  Restart error:", e$message, "\n")
  NULL
})

if (!is.null(fit_restart)) {
  ll_restart <- as.numeric(logLik(fit_restart$model))
  cat("  EM+Restart logLik:", ll_restart, "\n")
  rows[[idx]] <- data.frame(key = "loglik_em_restart", value = ll_restart); idx <- idx + 1
} else {
  rows[[idx]] <- data.frame(key = "loglik_em_restart", value = NA); idx <- idx + 1
}

out_fit <- do.call(rbind, rows)
write.csv(out_fit, file.path(outdir, "ref_advanced_fit.csv"), row.names = FALSE)
cat("Written: ref_advanced_fit.csv\n")

# ======================================================================
# 5. Formula-based MHMM
# ======================================================================
cat("\n=== Formula-based MHMM ===\n")

# Cluster 1 parameters
init_f1 <- c(0.7, 0.3)
trans_f1 <- matrix(c(0.8, 0.2, 0.3, 0.7), nrow = 2, byrow = TRUE)
emiss_f1 <- matrix(c(0.6, 0.3, 0.1, 0.1, 0.4, 0.5), nrow = 2, byrow = TRUE)

# Cluster 2 parameters
init_f2 <- c(0.4, 0.6)
trans_f2 <- matrix(c(0.6, 0.4, 0.2, 0.8), nrow = 2, byrow = TRUE)
emiss_f2 <- matrix(c(0.1, 0.3, 0.6, 0.5, 0.4, 0.1), nrow = 2, byrow = TRUE)

mhmm_formula <- tryCatch({
  build_mhmm(
    observations = seq_obj_10,
    initial_probs = list(init_f1, init_f2),
    transition_probs = list(trans_f1, trans_f2),
    emission_probs = list(emiss_f1, emiss_f2),
    formula = ~ x,
    data = covariate_data,
    cluster_names = c("Cluster1", "Cluster2")
  )
}, error = function(e) {
  cat("  Formula MHMM build error:", e$message, "\n")
  NULL
})

rows_f <- list()
idx_f <- 1

if (!is.null(mhmm_formula)) {
  ll_formula <- as.numeric(logLik(mhmm_formula))
  cat("  Formula MHMM logLik:", ll_formula, "\n")
  rows_f[[idx_f]] <- data.frame(key = "loglik_formula_before",
                                value = ll_formula); idx_f <- idx_f + 1

  # Fit with EM
  fit_formula <- tryCatch({
    fit_model(mhmm_formula,
              em_step = TRUE, global_step = FALSE, local_step = FALSE,
              control_em = list(maxeval = 1000, reltol = 1e-10,
                                print_level = 0))
  }, error = function(e) {
    cat("  Formula MHMM fit error:", e$message, "\n")
    NULL
  })

  if (!is.null(fit_formula)) {
    ll_formula_after <- as.numeric(logLik(fit_formula$model))
    cat("  Formula MHMM logLik after EM:", ll_formula_after, "\n")
    rows_f[[idx_f]] <- data.frame(key = "loglik_formula_after",
                                  value = ll_formula_after); idx_f <- idx_f + 1

    n_params_f <- attr(logLik(fit_formula$model), "df")
    cat("  Formula MHMM n_params:", n_params_f, "\n")
    rows_f[[idx_f]] <- data.frame(key = "n_params_formula",
                                  value = n_params_f); idx_f <- idx_f + 1
  }
} else {
  rows_f[[idx_f]] <- data.frame(key = "status",
                                value = "formula_build_failed"); idx_f <- idx_f + 1
}

out_formula <- do.call(rbind, rows_f)
write.csv(out_formula, file.path(outdir, "ref_advanced_formula.csv"),
          row.names = FALSE)
cat("Written: ref_advanced_formula.csv\n")

# ======================================================================
# 6. trim_model
# ======================================================================
cat("\n=== trim_model ===\n")

hmm_totrim <- build_hmm(observations = seq_obj,
                         transition_probs = trans_A,
                         emission_probs = emiss_A,
                         initial_probs = init_A)
fitted_totrim <- fit_model(hmm_totrim,
                           em_step = TRUE, global_step = FALSE,
                           local_step = FALSE,
                           control_em = list(maxeval = 1000, reltol = 1e-10,
                                             print_level = 0))

ll_before_trim <- as.numeric(logLik(fitted_totrim$model))
cat("  logLik before trim:", ll_before_trim, "\n")

trimmed <- tryCatch({
  trim_model(fitted_totrim$model, maxit = 100, zerotol = 1e-8)
}, error = function(e) {
  cat("  trim_model error:", e$message, "\n")
  NULL
})

rows_t <- list()
idx_t <- 1
rows_t[[idx_t]] <- data.frame(key = "loglik_before_trim",
                              value = ll_before_trim); idx_t <- idx_t + 1

if (!is.null(trimmed)) {
  ll_after_trim <- as.numeric(logLik(trimmed))
  cat("  logLik after trim:", ll_after_trim, "\n")
  rows_t[[idx_t]] <- data.frame(key = "loglik_after_trim",
                                value = ll_after_trim); idx_t <- idx_t + 1

  # Count zeros in trimmed model
  ip <- tryCatch(initial_probs(trimmed), error = function(e) trimmed$initial_probs)
  tp <- tryCatch(transition_probs(trimmed), error = function(e) trimmed$transition_probs)
  ep <- tryCatch(emission_probs(trimmed), error = function(e) trimmed$emission_probs)
  if (is.list(ep) && !is.matrix(ep)) ep <- as.matrix(ep[[1]])

  n_zeros_ip <- sum(as.numeric(ip) == 0)
  n_zeros_tp <- sum(as.matrix(tp) == 0)
  n_zeros_ep <- sum(as.matrix(ep) == 0)
  cat("  Zeros in initial_probs:", n_zeros_ip, "\n")
  cat("  Zeros in transition_probs:", n_zeros_tp, "\n")
  cat("  Zeros in emission_probs:", n_zeros_ep, "\n")

  rows_t[[idx_t]] <- data.frame(key = "n_zeros_initial",
                                value = n_zeros_ip); idx_t <- idx_t + 1
  rows_t[[idx_t]] <- data.frame(key = "n_zeros_transition",
                                value = n_zeros_tp); idx_t <- idx_t + 1
  rows_t[[idx_t]] <- data.frame(key = "n_zeros_emission",
                                value = n_zeros_ep); idx_t <- idx_t + 1
} else {
  rows_t[[idx_t]] <- data.frame(key = "loglik_after_trim",
                                value = NA); idx_t <- idx_t + 1
}

out_trim <- do.call(rbind, rows_t)
write.csv(out_trim, file.path(outdir, "ref_advanced_trim.csv"),
          row.names = FALSE)
cat("Written: ref_advanced_trim.csv\n")

# ======================================================================
# 7. Model comparison: BIC across different n_states
# ======================================================================
cat("\n=== Model comparison ===\n")
rows_c <- list()
idx_c <- 1

for (ns in c(2, 3, 4)) {
  cat("  n_states =", ns, "\n")
  hmm_ns <- tryCatch({
    build_hmm(observations = seq_obj, n_states = ns)
  }, error = function(e) {
    cat("    build error:", e$message, "\n")
    NULL
  })

  if (!is.null(hmm_ns)) {
    set.seed(42)
    fit_ns <- tryCatch({
      fit_model(hmm_ns, em_step = TRUE, global_step = FALSE,
                local_step = FALSE,
                control_em = list(maxeval = 1000, reltol = 1e-10,
                                  print_level = 0))
    }, error = function(e) {
      cat("    fit error:", e$message, "\n")
      NULL
    })

    if (!is.null(fit_ns)) {
      ll_ns <- as.numeric(logLik(fit_ns$model))
      aic_ns <- AIC(fit_ns$model)
      bic_ns <- BIC(fit_ns$model)
      df_ns <- attr(logLik(fit_ns$model), "df")
      cat("    logLik:", ll_ns, "AIC:", aic_ns, "BIC:", bic_ns,
          "df:", df_ns, "\n")

      rows_c[[idx_c]] <- data.frame(key = paste0("loglik_", ns, "states"),
                                    value = ll_ns); idx_c <- idx_c + 1
      rows_c[[idx_c]] <- data.frame(key = paste0("aic_", ns, "states"),
                                    value = aic_ns); idx_c <- idx_c + 1
      rows_c[[idx_c]] <- data.frame(key = paste0("bic_", ns, "states"),
                                    value = bic_ns); idx_c <- idx_c + 1
      rows_c[[idx_c]] <- data.frame(key = paste0("df_", ns, "states"),
                                    value = df_ns); idx_c <- idx_c + 1
    }
  }
}

if (length(rows_c) > 0) {
  out_comp <- do.call(rbind, rows_c)
  write.csv(out_comp, file.path(outdir, "ref_advanced_comparison.csv"),
            row.names = FALSE)
  cat("Written: ref_advanced_comparison.csv\n")
}

cat("\nAll advanced reference files written to:", outdir, "\n")
