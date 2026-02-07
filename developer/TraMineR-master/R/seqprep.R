## ===========================
## Treatment of missing values
## ===========================

seqprep <- function(seqdata, left=NA, right="DEL", gaps=NA,
	neutral="#", missing=NA, void="%", nr="*") {

	nbseq <- nrow(seqdata)
	sl <- ncol(seqdata)

	message(" [>] preparing ",nbseq, " sequences")
	message(" [>] coding void elements with '", void, "' and missing values with '", nr,"'")

	if (is.na(missing)) {
		mstate <- is.na(seqdata)
	}
	else {
		mstate <- seqdata==missing
	}

	allmiss <- NULL
	for (i in 1:nbseq) {
		nbmiss <- sum(mstate[i,], na.rm=TRUE)
#		if (nbmiss>0 && nbmiss<sl) {
#			seqdata[i,] <- TraMineR.trunc(seqdata=seqdata[i,], mstate=mstate[i,], sl=sl,
#				left=left, right=right, gaps=gaps,
#				neutral=neutral, void=void)
#		}
#		else
    if (nbmiss==sl) {
			allmiss <- c(allmiss,i)
		}
    if (nbmiss > 0){
			seqdata[i,] <- TraMineR.trunc(seqdata=seqdata[i,], mstate=mstate[i,], sl=sl,
				left=left, right=right, gaps=gaps,
				neutral=neutral, void=void)
    }
	}

  nempty <- length(allmiss)
	if (nempty>0) {
    dots <- ""
    if (nempty > 10){
      allmiss <- allmiss[1:10]
      dots <- ", ..."
    }
		message(" [!!] ",nempty," empty sequence(s) with index: ", paste(allmiss, collapse=","),dots,"\n      may produce inconsistent results.")
	}

	## Setting a new code for missing statuses
	if (is.na(missing)) seqdata[is.na(seqdata)] <- nr
	else seqdata[seqdata==missing] <- nr

	return(seqdata)
}
