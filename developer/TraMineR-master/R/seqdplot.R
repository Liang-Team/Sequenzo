## ========================
## State distribution plot
## ========================

seqdplot <- function(seqdata, group=NULL, main="auto", ...) {
	seqplot(seqdata, group=group, type="d", main=main, ...)
}

seqdHplot <- function(seqdata, group=NULL, main="auto", ...) {
	seqplot(seqdata, group=group, type="dH", main=main, ...)
}
