## =============================
## Plotting medoids of relative frequeny group of sequences
## =============================

seqrfplot <- function(seqdata, group = NULL, main = "auto", ...) {
	seqplot(seqdata, group=group, type="rf", main=main, ...)
}
