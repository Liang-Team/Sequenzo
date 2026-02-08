# @Author  : Yuqi Liang 梁彧祺
# @File    : traminer_twed_reference.R
# @Time    : 2026/02/07 21:40
# @Desc    :
# TraMineR TWED reference for cross-check with Sequenzo.
# Usage: Rscript traminer_twed_reference.R [outdir]
# Example: Rscript traminer_twed_reference.R .
# Requires: install.packages("TraMineR")

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("Install TraMineR: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."

# Same 4 sequences as in test_dissimilarity_measures_traminer.py (Part 3 TWED): (1,1,2,2,1), (1,2,2,1,1), (2,1,1,2,2), (1,1,1,2,2)
dat <- matrix(c(
  1,1,2,2,1,
  1,2,2,1,1,
  2,1,1,2,2,
  1,1,1,2,2
), nrow = 4, ncol = 5, byrow = TRUE)
colnames(dat) <- paste0("T", 1:5)
rownames(dat) <- c("s0","s1","s2","s3")
dat <- as.data.frame(dat)

seqdata <- TraMineR::seqdef(dat, var = paste0("T", 1:5), alphabet = c(1, 2))
sm <- matrix(c(0, 2, 2, 0), nrow = 2, ncol = 2)

# Base config (norm=none, nu=0.5, h=0.5, indel=2)
D <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none", sm = sm, indel = 2, nu = 0.5, h = 0.5)
write.csv(as.matrix(D), file.path(outdir, "ref_twed_none_05_05_indel2.csv"), row.names = TRUE)

# Part 3b configs: different nu, h, norm, indel
D_nu01_h05 <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none", sm = sm, indel = 2, nu = 0.1, h = 0.5)
write.csv(as.matrix(D_nu01_h05), file.path(outdir, "ref_twed_nu01_h05.csv"), row.names = TRUE)

D_nu1_h01 <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none", sm = sm, indel = 2, nu = 1.0, h = 0.1)
write.csv(as.matrix(D_nu1_h01), file.path(outdir, "ref_twed_nu1_h01.csv"), row.names = TRUE)

D_nu1_h1 <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none", sm = sm, indel = 2, nu = 1.0, h = 1.0)
write.csv(as.matrix(D_nu1_h1), file.path(outdir, "ref_twed_nu1_h1.csv"), row.names = TRUE)

D_yujianbo <- TraMineR::seqdist(seqdata, method = "TWED", norm = "YujianBo", sm = sm, indel = 2, nu = 0.5, h = 0.5)
write.csv(as.matrix(D_yujianbo), file.path(outdir, "ref_twed_yujianbo.csv"), row.names = TRUE)

D_indel1 <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none", sm = sm, indel = 1, nu = 0.5, h = 0.5)
write.csv(as.matrix(D_indel1), file.path(outdir, "ref_twed_indel1.csv"), row.names = TRUE)

cat("TWED reference matrices written to", outdir, "\n")
