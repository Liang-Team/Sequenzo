if (!requireNamespace("seqHMM", quietly = TRUE)) {
  stop("seqHMM required: install.packages('seqHMM')")
}
if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(TraMineR)
  library(seqHMM)
})

ids <- paste0("s", 0:3)
ch1_data <- data.frame(
  t1 = c("A", "A", "B", "B"),
  t2 = c("A", "A", "B", "B"),
  t3 = c("A", "B", "B", "A"),
  t4 = c("A", "A", "B", "B"),
  stringsAsFactors = FALSE
)
ch2_data <- data.frame(
  t1 = c("X", "X", "Y", "Y"),
  t2 = c("X", "X", "Y", "Y"),
  t3 = c("X", "Y", "Y", "X"),
  t4 = c("X", "X", "Y", "Y"),
  stringsAsFactors = FALSE
)

ch1 <- seqdef(ch1_data, alphabet = c("A", "B"), id = ids)
ch2 <- seqdef(ch2_data, alphabet = c("X", "Y"), id = ids)

model <- build_mhmm(
  observations = list(ch1, ch2),
  initial_probs = list(c(0.95, 0.05), c(0.95, 0.05)),
  transition_probs = list(
    matrix(c(0.90, 0.10, 0.10, 0.90), nrow = 2, byrow = TRUE),
    matrix(c(0.90, 0.10, 0.10, 0.90), nrow = 2, byrow = TRUE)
  ),
  emission_probs = list(
    list(
      matrix(c(0.95, 0.05, 0.10, 0.90), nrow = 2, byrow = TRUE),
      matrix(c(0.88, 0.12, 0.25, 0.75), nrow = 2, byrow = TRUE)
    ),
    list(
      matrix(c(0.05, 0.95, 0.90, 0.10), nrow = 2, byrow = TRUE),
      matrix(c(0.12, 0.88, 0.80, 0.20), nrow = 2, byrow = TRUE)
    )
  ),
  cluster_names = c("mostly-AX", "mostly-BY"),
  channel_names = c("letter", "marker")
)

write.csv(
  data.frame(key = "loglik", value = as.numeric(logLik(model))),
  file.path(outdir, "ref_mhmm_multichannel_loglik.csv"),
  row.names = FALSE
)

post <- as.data.frame(posterior_cluster_probabilities(model))
write.csv(
  post,
  file.path(outdir, "ref_mhmm_multichannel_posterior_cluster.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(cluster = as.character(most_probable_cluster(model, type = "posterior"))),
  file.path(outdir, "ref_mhmm_multichannel_cluster.csv"),
  row.names = FALSE
)
