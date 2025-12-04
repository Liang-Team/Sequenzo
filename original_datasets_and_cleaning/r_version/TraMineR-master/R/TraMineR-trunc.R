## SEQUENCE TRUNCATION

TraMineR.trunc <- function(seqdata, mstate, sl, left = "DEL", right = "DEL",
  gaps = "DEL", neutral = "#", void = "%") {

	sidx <- 1:sl

	## Index des missing et index des etats valides
	na.pos <- sidx[mstate]
	notna.pos <- sidx[!mstate]
  lc <- 0

  if(length(notna.pos)<1) { ## only missing values in sequence
    c1 <- 0
    rc <- 1
    mm <- NULL # most probably not used
  }
  else {

  	## Position of first valid state
  	c1 <- notna.pos[1]

  	if (c1>1)
  		lc <- c1-1
  	# else lc=0

  	rc <- max(notna.pos)+1
  	mm <- na.pos[na.pos > lc+1 & na.pos < rc-1]
  }

	seqdata.trunc <- seqdata

	if (!is.na(left) & lc>0) {
		if (left=="DEL") seqdata.trunc[1:lc] <- void
		else if (left=="NEUTRAL") seqdata.trunc[1:lc] <- neutral
		else seqdata.trunc[1:lc] <- left
	}

	if (!is.na(right) & rc<=sl) {
		if (right=="DEL") seqdata.trunc[rc:sl] <- void
		else if (right=="NEUTRAL") seqdata.trunc[rc:sl] <- neutral
		else seqdata.trunc[rc:sl] <- right
	}

	if (!is.na(gaps) & length(mm>0)) {
		if (gaps=="DEL") seqdata.trunc[mm] <- void
		else if (gaps=="NEUTRAL") seqdata.trunc[mm] <- neutral
		else seqdata.trunc[mm] <- gaps
	}

	ndel <- sum(seqdata.trunc==void, na.rm=TRUE)

	if (ndel>0) {
		seqdata.trunc <- seqdata.trunc[seqdata.trunc!=void]
		seqdata.trunc <- c(seqdata.trunc,rep(void,ndel))
	}

	return(seqdata.trunc)
}
