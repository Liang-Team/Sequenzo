#!/usr/bin/env Rscript
# ============================================================
# TraMineR Benchmark: OM, LCS, LCP, EUCLID
# Usage: Rscript run_traminer.R <dataset_dir>
# Example: Rscript run_traminer.R generated_datasets/output_n10000_l30_u85
# ============================================================

library(TraMineR)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript run_traminer.R <dataset_dir>\n")
  cat("Example: Rscript run_traminer.R generated_datasets/output_n10000_l30_u85\n")
  quit(status = 1)
}

dataset_dir <- args[1]
if (!dir.exists(dataset_dir)) {
  cat(sprintf("Error: directory not found: %s\n", dataset_dir))
  quit(status = 1)
}

# Find .dat or .csv files
fnames <- sort(c(
  list.files(dataset_dir, pattern = "\\.dat$", full.names = TRUE),
  list.files(dataset_dir, pattern = "\\.csv$", full.names = TRUE)
))
if (length(fnames) == 0) {
  cat(sprintf("Error: no .dat or .csv files in %s\n", dataset_dir))
  quit(status = 1)
}

# ---- Data loading (same logic as Python scripts) ----
pd_list <- list()
total_ids <- 0
for (fname in fnames) {
  ldata <- read.csv(fname)
  ldata$id <- ldata$id + total_ids
  total_ids <- max(ldata$id) + 1
  pd_list[[length(pd_list) + 1]] <- ldata
}
data <- do.call(rbind, pd_list)

# ---- Convert spell format to wide format ----
id_list <- sort(unique(data$id))
t_min <- min(data$stime)
t_max <- max(data$etime)
time_points <- t_min:t_max

# Build wide matrix (rows=ids, cols=time points)
pdata <- matrix(NA, nrow = length(id_list), ncol = length(time_points))
rownames(pdata) <- id_list
colnames(pdata) <- time_points

id_idx <- match(data$id, id_list)
for (i in seq_len(nrow(data))) {
  row_idx <- id_idx[i]
  for (d in data$stime[i]:data$etime[i]) {
    col_idx <- d - t_min + 1
    if (col_idx >= 1 && col_idx <= ncol(pdata)) {
      pdata[row_idx, col_idx] <- data$event[i]
    }
  }
}

pdata_df <- as.data.frame(pdata)

# ---- Create sequence object ----
seq_obj <- seqdef(pdata_df, id = id_list, xtstep = 1)

n_seq <- nrow(pdata_df)
n_time <- ncol(pdata_df)
dataset_name <- basename(dataset_dir)

cat(sprintf("Dataset: %s, n=%d, time_points=%d\n", dataset_name, n_seq, n_time))
cat(strrep("=", 60), "\n")

# ---- Substitution cost matrix (TRATE, matching Sequenzo's default) ----
sm_trate <- seqsubm(seq_obj, method = "TRATE")
indel_auto <- max(sm_trate) / 2

# ---- Benchmark each method ----
methods <- c("OM", "LCS", "LCP", "EUCLID")

for (m in methods) {
  cat(sprintf("[TraMineR] Running %s ... ", m))

  t_start <- proc.time()

  if (m == "OM") {
    dist_matrix <- seqdist(seq_obj, method = "OM", sm = sm_trate, indel = indel_auto)
  } else if (m == "LCS") {
    dist_matrix <- seqdist(seq_obj, method = "LCS")
  } else if (m == "LCP") {
    dist_matrix <- seqdist(seq_obj, method = "LCP")
  } else if (m == "EUCLID") {
    dist_matrix <- seqdist(seq_obj, method = "EUCLID")
  }

  t_elapsed <- (proc.time() - t_start)["elapsed"]
  cat(sprintf("time_elapsed: %.4f\n", t_elapsed))
}

cat(strrep("=", 60), "\n")
