## =========================
## Sequences frequency table
## =========================

seqstatf <- function(seqdata, weighted=TRUE, with.missing=FALSE) {

	if (!is.stslist(seqdata))
		stop("data is not a sequence object, use seqdef function to create one")

	istatd <- suppressMessages(seqistatd(seqdata, with.missing=with.missing))

	## Weights
	weights <- attr(seqdata, "weights")

	if (!weighted || is.null(weights))
		weights <- rep(1, nrow(seqdata))

	istatd <- istatd*weights

	Freq <- colSums(istatd)

	Percent <- Freq/sum(Freq)*100

	res <- data.frame(Freq, Percent)

	return(res)

}
