## =============================================
## Number of distinct subsequences in a sequence
## =============================================

nsubs <- function (x, nbstat, statlist, void, nr, with.missing) {
		l <- vector(mode="integer", nbstat)
		x <- x[x!=void]
    if (!with.missing) x <- x[x!=nr]
		slength <- length(x)
    if (slength == 0) return(1) ## empty sequence has one subseq (the empty one)

		N <- vector(mode="integer",(slength+1))
		N[1] <- 1

		for (i in 2:(slength+1)) {
			N[i] <- 2*N[i-1]
			cidx <- which(statlist==x[i-1])

			if (l[cidx[1]] > 0) N[i] <- N[i] - N[l[cidx[1]]]
			l[cidx[1]] <- i-1
		}
		return(N[(slength+1)])
	}


seqsubsn <- function(seqdata, DSS=TRUE, with.missing=FALSE) {

	if (!inherits(seqdata,"stslist"))
		stop("data is not a sequence object, see seqdef function to create one")

## Since v 2.0.13 we use thw with.missing argument
	## with.missing=FALSE
	##nr <- attr(seqdata,"nr")
	##if (any(seqdata==nr)) {
	##	message(" [!] found missing state in the sequence(s), adding missing state to the alphabet")
	##	with.missing=TRUE
	##}

	if (DSS==TRUE) {
		seqdata <- suppressMessages(seqdss(seqdata, with.missing=with.missing))
	}

	## alphabet
	#sl <- attr(seqdata,"alphabet")
	#if (with.missing) {
	#	sl <- c(sl, nr)
	#}
  sl <- alphabet(seqdata, with.missing=with.missing)

	ns <- length(sl)

	void <- attr(seqdata,"void")
	nr <- attr(seqdata,"nr")
		
	result <- apply(seqdata, 1, nsubs, nbstat=ns, statlist=sl, void=void, nr=nr, with.missing=with.missing)

	result <- as.matrix(result)
	colnames(result) <- "Subseq."
	rownames(result) <- rownames(seqdata)

	return(result)
}
