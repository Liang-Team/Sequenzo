# TraMineRextras seqsamm() toy reference — run from repo root:
#   Rscript tests/with_event_history_analysis/seqsamm_toy_reference.R
#
# Compare printed summaries with tests/with_event_history_analysis/test_seqsamm_toy_scenarios.py

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("Install TraMineR: install.packages('TraMineR')")
}
if (!requireNamespace("TraMineRextras", quietly = TRUE)) {
  stop("Install TraMineRextras: install.packages('TraMineRextras')")
}

library(TraMineR)
library(TraMineRextras)

cat("\n=== 1. No void (numeric states, L=5, sublength=2) ===\n")
m1 <- matrix(
  c(1, 2, 3, 1, 2,
    2, 3, 1, 2, 3,
    3, 1, 2, 3, 1),
  nrow = 3, byrow = TRUE
)
seq1 <- seqdef(m1, alphabet = 1:3, states = 1:3, labels = c("A", "B", "C"))
s1 <- seqsamm(seq1, sublength = 2)
cat("nrow:", nrow(s1), "\n")
cat("time range:", range(s1$time), "\n")
cat("spell.time summary:\n")
print(summary(s1$spell.time))
print(head(s1[, c("id", "time", "spell.time", "transition", "s.1", "s.2")], 12))

cat("\n=== 2. With void (% padding, L=4, sublength=2) ===\n")
m2 <- matrix(
  c("%", "%", "A", "B",
    "%", "A", "A", "B"),
  nrow = 2, byrow = TRUE
)
seq2 <- seqdef(m2, alphabet = c("%", "A", "B"), states = c("%", "A", "B"))
s2 <- seqsamm(seq2, sublength = 2)
cat("nrow:", nrow(s2), "\n")
print(s2[, c("id", "time", "spell.time", "transition", "s.1", "s.2")])

cat("\n=== 3. With covar ===\n")
covar <- data.frame(x = c(10, 20, 30), row.names = 1:3)
s3 <- seqsamm(seq1, sublength = 2, covar = covar)
cat("nrow:", nrow(s3), "\n")
cat("x by id:\n")
print(tapply(s3$x, s3$id, function(v) v[1]))
