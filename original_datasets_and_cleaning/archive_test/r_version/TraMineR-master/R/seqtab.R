## =========================
## Sequences frequency table
## =========================

seqtab <- function(seqdata, idxs = 1:10, weighted = TRUE, format = "SPS", tlim) {

  TraMineR.check.depr.args(alist(idxs = tlim))

	if (!inherits(seqdata,"stslist"))
		stop("data is not a sequence object, use seqdef function to create one")

  if(is.null(idxs) || (length(idxs)>1 && min(idxs)<1) || any(idxs<0))
		stop("idxs should be a non negative integer or a strictly positive vector.")


	## Eliminating empty sequences
	seqdata <- seqdata[rowSums(seqdata!=attr(seqdata,"nr"))!=0,]

	## Weights
	weights <- attr(seqdata, "weights")

	if (!weighted || is.null(weights)) {
		weights <- rep(1.0, nrow(seqdata))
	}
	## Also takes into account that in unweighted sequence objects created with
	## older TraMineR versions the weights attribute is a vector of 1
	## instead of NULL
	if (all(weights==1))
		weighted <- FALSE

	if (seqfcheck(seqdata)=="-X")
		warning("'-' character in state codes may cause invalid results")

	if (format=="SPS") {
		seqlist <- suppressMessages(seqformat(seqdata, from = "STS", to = "SPS",
		  SPS.out = list(xfix = "", sdsep = "/"), compress = TRUE))
	}
	else if (format=="STS")
		seqlist <- seqconc(seqdata)
	else
		stop("Format must be one of: STS or SPS")

	Freq <- tapply(weights, seqlist, sum)

	Freq <- sort(Freq, decreasing=TRUE)
	Percent <- Freq/sum(Freq)*100

	nbuseq <- length(Freq)

  if (idxs[1]==0 || max(idxs)>nbuseq) {
		idxs <- 1:nbuseq
	}

	##if (nbuseq >1){
        idxf <- match(names(Freq), seqlist)[idxs]
        res <- seqdata[idxf,]
    ##}
    ##else {
    ##    res <- names(Freq)
    ##}
	table <- data.frame(Freq, Percent)[idxs,]

	## ==================================
	## DEFINING CLASS AND SOME ATTRIBUTES
	## ==================================
	class(res) <- c("stslist.freq",class(res))

	## Setting the weights of the object equal to the frequencies
	attr(res, "weights") <- table$Freq

	attr(res,"freq") <- table
	attr(res,"nbseq") <- sum(weights)
	attr(res,"weighted") <- weighted
	attr(res,"idxs") <- idxs
	attr(res,"idxf") <- idxf
	attr(res,"format") <- format

	return(res)
}
