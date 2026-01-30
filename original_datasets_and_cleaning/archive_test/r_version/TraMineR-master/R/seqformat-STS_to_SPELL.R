# Should only be used through seqformat()

## =======================================
## Extracts distinct states from sequences
## =======================================

STS_to_SPELL <- function(seqdata, pdata = NULL, pvar = NULL, with.missing = TRUE) {

	if (!inherits(seqdata,"stslist")){
		stop("data is NOT a state sequence object, see seqdef function to create one")
	}

	nseqs <- nrow(seqdata)

	sl <- seqlength(seqdata)
	sltot <- sum(sl)

	void <- attr(seqdata, "void")
	statl <- attr(seqdata, "alphabet")
	nr <- attr(seqdata, "nr")

	if (is.data.frame(pdata)) {
    if (inherits(pdata[,pvar[2]], "Date") || is.character(pdata[,pvar[2]]) || is.factor(pdata[,pvar[2]]))
      stop(" [!] Column ", pvar[2]," of pdata should contain integer (birth year) values", call.=FALSE)
	  pids <- pdata[, pvar[1]]
	  pbirths <- pdata[, pvar[2]] - 1
	  if (length(pids) != nseqs)
	    msg.stop("'pdata' id column must contain one entry per sequence")
	  if (length(pbirths) != nseqs)
	    msg.stop("'pdata' birth column must contain one entry per sequence")
	} else {
	  pids <- rownames(seqdata, do.NULL = FALSE, prefix = "")
	  pbirths <- rep(0, nseqs)
	}

	begin <- numeric(sltot)
	end <-  numeric(sltot)
	ids <- vector(mode = mode(pids), length = sltot)
	states <- character(sltot)
	if(with.missing) {
		statl <- c(statl, nr)
	}

	seqdatamat <- as.matrix(seqdata)

	if (!with.missing){
		seqdatamat[seqdatamat==nr] <- void
	}
	itot <- 1
	for (i in 1:nseqs) {

		idx <- 1
		sli <- sl[i]-1
		while (idx <= sl[i]) {
			ids[itot] <- pids[i]
			iseq <- seqdatamat[i, idx]
			begin[itot] <- pbirths[i] + idx
			# if(itot ==1){
				# print(iseq)
				# print(str(states))
			# }
			while (idx <= sli && (seqdatamat[i, idx+1]==iseq)) {
				idx <- idx+1
			}

			if (iseq != void) {
				states[itot] <- as.character(iseq)
				end[itot] <- pbirths[i] + idx
				# if(itot ==1){
					# print(head(spell))
				# }
				itot <- itot+1
			}
			idx <- idx+1
		}

	}
	## drop=FALSE ensures that the result is a matrix even if trans has only one row
	keep <- 1:(itot-1)
	spell <- data.frame(id=ids[keep], begin=begin[keep], end=end[keep], states=factor(states[keep], levels=statl))

	return(spell)
}
