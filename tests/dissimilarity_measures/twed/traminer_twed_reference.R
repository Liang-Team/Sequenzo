# @Author  : Yuqi Liang 梁彧祺
# @File    : traminer_twed_reference.R
# @Time    : 2026/02/07 21:40
# @Desc    : 
# TraMineR TWED reference output for cross-check with Sequenzo.
# Run from repo root: Rscript tests/traminer_twed_reference.R
# Requires: install.packages("TraMineR")

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("Install TraMineR: install.packages('TraMineR')")
}

# Same 4 sequences as in test_twed_traminer.py: (1,1,2,2,1), (1,2,2,1,1), (2,1,1,2,2), (1,1,1,2,2)
# TraMineR seqdef uses alphabet as given; seqasnum then gives 0-based codes (0, 1 for two states)
dat <- matrix(c(
  1,1,2,2,1,
  1,2,2,1,1,
  2,1,1,2,2,
  1,1,1,2,2
), nrow = 4, ncol = 5, byrow = TRUE)
colnames(dat) <- paste0("T", 1:5)
rownames(dat) <- c("s0","s1","s2","s3")
dat <- as.data.frame(dat)

# State sequence object (TraMineR uses 1-based labels in display; internal codes 0-based in C)
seqdata <- TraMineR::seqdef(dat, var = paste0("T", 1:5), alphabet = c(1, 2))

# Substitution matrix: 2x2 for states 1,2 -> in C 0,1 so sm[1,1]=0, sm[1,2]=sm[2,1]=2, sm[2,2]=0
# In R matrix by column: c(0,2,2,0) -> [0 2; 2 0]
sm <- matrix(c(0, 2, 2, 0), nrow = 2, ncol = 2)

# TWED with norm="none", nu=0.5, h=0.5 (lambda), sm as above
# TraMineR seqdist expects sm to be the full matrix for the alphabet (R uses 0-based in C)
D <- TraMineR::seqdist(seqdata, method = "TWED", norm = "none",
  sm = sm, indel = 2, nu = 0.5, h = 0.5)

# Print as matrix for copy-paste (dist object has Labels attribute)
Dmat <- as.matrix(D)
cat("TraMineR TWED (norm=none, nu=0.5, h=0.5, sm=[[0,2],[2,0]], indel=2):\n")
print(round(Dmat, 10))
cat("\nUpper triangle (0-based indices) for programmatic comparison:\n")
for (i in 1:3) for (j in (i+1):4) cat(sprintf("  D[%d,%d]=%.10f\n", i-1, j-1, Dmat[i,j]))
