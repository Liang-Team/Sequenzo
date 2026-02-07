## ========================================
## Concatenates vectors of states or events
## into character strings
## ========================================

sconc <- function(seqdata, sep, void) {

	if (is.na(void))
		vi <- !is.na(seqdata)
	else if (!is.null(void))
		vi <- seqdata!=void
	else vi <- 1:length(seqdata)

	return(paste(seqdata[vi], collapse=sep))
	}

seqconc <- function (data, var=NULL, sep="-", vname="Sequence", void=NA) {

	if (inherits(data,"stslist")) {
		void <- attr(data,"void")
		cseq <- apply(data, 1, sconc, sep, void)
		cseq <- as.matrix(cseq)
		rownames(cseq) <- rownames(data)
	}
	else {
		seqdata <- seqxtract(data, var)

		if (seqdim(seqdata)[1]==1)
			cseq <- sconc(seqdata,sep, void)
		else
			cseq <- apply(seqdata, 1, sconc, sep, void)

		cseq <- as.matrix(cseq)

		## Rows and column names for the output
    if (is.null(rownames(data))){
		  rownames(cseq) <- paste("[",seq(1:length(cseq)),"]",sep="")
    } else {
      rownames(cseq) <- rownames(data)
    }
	}

	colnames(cseq) <- vname

	return(cseq)
}
