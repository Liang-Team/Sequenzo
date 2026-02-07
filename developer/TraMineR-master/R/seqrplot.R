## ============================================
## PLOT A REPRESENTATIVE SEQUENCE
## ============================================

seqrplot <- function(seqdata, group = NULL, main = "auto", ...) {
	seqplot(seqdata, group=group, type="r", main=main, ...)
}
