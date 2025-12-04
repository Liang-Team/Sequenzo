library(reticulate)
library(TraMineR)

pickle   <- import("pickle")
builtins <- import_builtins()

#setwd("./Sequenzo Issue")

# Open and load the pickle file
con    <- builtins$open("dataset.pkl", "rb")
loaded <- pickle$load(con)
con$close()

# Extract X and y
X_py <- loaded[["X"]]  # numpy array (N, T, C)
y_py <- loaded[["y"]]  # numpy array (N,)

# Convert to R objects
X <- py_to_r(X_py)
y <- as.integer(py_to_r(y_py))

cat("Shape X:", dim(X), "\n")  # (N, T, C)
cat("Shape y:", length(y), "\n")

N <- dim(X)[1]
Tlen <- dim(X)[2]
C <- dim(X)[3]

# ---------- 2) Construction of TraMineR multichannel sequences ----------
# We create a list of C seqdef objects, one for each binary channel

# ---------- 2) Subsample: 10 sequences ----------
n_sub <- min(10, N)
idx   <- seq_len(n_sub)  # here: the first 10
cat("Working on", n_sub, "sequences (indices:", idx, ")\n")

X_sub <- X[idx, , , drop = FALSE]  # (n_sub, T, C)
y_sub <- y[idx]

# ---------- 3) Construction of TraMineR multichannel sequences ----------
seq.list <- vector("list", C)

for (cc in seq_len(C)) {
  # X_sub[,,cc]: matrix n_sub x Tlen for channel cc
  mat <- X_sub[ , , cc]
  
  df <- as.data.frame(mat)
  colnames(df) <- paste0("t", seq_len(Tlen))
  
  # binary 0/1 -> factor
  df[] <- lapply(df, function(col) factor(col, levels = c(0, 1)))
  
  seq.list[[cc]] <- seqdef(df)
}

# ---------- 3) Multivariate OM distance with TRATE costs ----------
# TRATE = “transition rates” for substitution costs
# (Multichannel OM, insertion/deletion cost = 1 by default)

dist.om <- seqMD(seq.list,
                 method = "OM",
                 sm = "TRATE", 
                 what = "diss")

class(dist.om)

# ---------- 4) Save dist.om as pickle ----------

pickle <- import("pickle")
builtins <- import_builtins()

# Convert the "dist" object to an R matrix
Dmat <- as.matrix(dist.om)

# Open a pickle file in binary write mode
con <- builtins$open("dist_om_traminer.pkl", "wb")

# Save the matrix to the file
pickle$dump(Dmat, con)

# Close the file
con$close()

cat("File 'dist_om_traminer.pkl' saved.\n")
