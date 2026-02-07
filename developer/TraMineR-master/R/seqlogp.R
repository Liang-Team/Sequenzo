seqlogp <- function(seqdata, prob="trate", time.varying=TRUE, begin="freq", weighted=TRUE, with.missing=FALSE) {
	## Liste des taux de transitions par age

  if (!isTRUE(with.missing) && any(seqdata==attr(seqdata,"nr"))) {
    stop("Non-void missing values in seqdata, with.missing should be TRUE!")
  }

	sl <- seqlength(seqdata, with.missing=with.missing)
	maxage <- max(sl)
	nbtrans <- maxage -1
	agedtr <- vector(mode="list", length=maxage)
	
	## On ajoute 1 pour que les codes correspondent aux index R (commence à 1)
	seqdatanum <- seqasnum(seqdata, with.missing=with.missing)+1
	nbstates <- max(seqdatanum, na.rm=TRUE)
	## User defined begin frequencies
	if(is.numeric(begin)){
		if (length(begin)!=nbstates) {
			stop("Begin frequencies should be a numeric vector of length ", nbstates)
		}
		message(" [>] Using user defined frequencies as starting point")
		firstfreq <- begin
	}
	##Compute from data
	else if (is.character(begin) && begin=="freq") {
		message(" [>] Using frequencies at first position as starting point")
		firstfreq <- seqstatd(seqdata[,1:2], weighted=weighted, with.missing=with.missing)$Frequencies[, 1]
	}
	else if (is.character(begin) && begin=="global.freq") {
		message(" [>] Using overall state frequencies as starting point")
		firstfreq <- seqstatf(seqdata, weighted=weighted, with.missing=with.missing)$Percent/100
  }
  else {
		stop("Unknow method to compute starting frequencies")
	}
	
	###Automatic method to compute transition rates
	if (is.character(prob)) {
		if (prob=="trate") {
			if (time.varying) {
				message(" [>] Using time varying transition rates as probability model")
				agedtr <- suppressMessages(seqtrate(seqdata, time.varying=TRUE, weighted=weighted, with.missing=with.missing))
			}
			else {
				message(" [>] Using global transition rates as probability model")
				agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
				tr <- suppressMessages(seqtrate(seqdata, weighted=weighted, with.missing=with.missing))
				for (i in 1:nbtrans) {
					agedtr[,,i] <- tr
				}
			}
		}
		else if (prob=="freq") {
			## building a transition matrix that does not depend on previous state
			## so that same algorithm can be used
			message(" [>] Using time varying frequencies as probability model")
			agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
			if (time.varying) {
				freqs <- seqstatd(seqdata, weighted=weighted, with.missing=with.missing)$Frequencies
				for (i in 1:nbtrans) {
					for (j in 1:length(freqs[, i+1])) {
							agedtr[, j,i] <- freqs[j, i+1]
          }
				}
			}
			else {
				message(" [>] Using global frequencies as probability model")
				freqs <- seqstatf(seqdata, weighted=weighted, with.missing=with.missing)$Percent/100
				for (i in 1:nbtrans) {
					for (j in 1:length(freqs)) {
   					agedtr[, j,i] <- freqs[j]
	  			}
				}
			}
		}
		else {
			stop("Unknow method to compute transition rate")
		}
	}
	## User defined transition rates
	else{
		if(is.array(prob)){
			if(length(dim(prob)) == 3) {
				##Correct dimensions
				if(any(dim(prob)!=c(nbstates, nbstates, nbtrans))){
					stop("Transition rate should be an array of size (state x state x transition) ",
						nbstates,"x", nbstates, "x", nbtrans)
				}
				message(" [>] Using user defined time varying transition rates as probability model")
				agedtr <- prob
			} else if (length(dim(prob)) == 2) {
				message(" [>] Using user defined global transition rates as probability model")
				if(any(dim(prob)!=c(nbstates, nbstates))){
					stop("Transition rate should be a matrix of size (state x state) ",
						nbstates,"x", nbstates)
				}
				agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
				for (i in 1:nbtrans) {
					agedtr[,,i] <- prob
				}
			}
			else {
				stop("Transition rate should be an array of size (state x state x transition) ",
						nbstates,"x", nbstates, "x", nbtrans, " or a matrix of size (state x state) ",
						nbstates,"x", nbstates)
			}
		}
		else {
			stop("Unknow method to compute transition rate")
		}
	}
	logp <- numeric(length=(nrow(seqdata)))
	logp[] <- 0
	for (i in 1:nrow(seqdatanum)) {
		p <- firstfreq[seqdatanum[i, 1]]
    if (!is.na(p) & p > 0) 	logp[[i]] <- -log(p)
		if (sl[i]>1) {
			for (j in 2:sl[i]) {
				p <- agedtr[seqdatanum[i, j-1], seqdatanum[i, j], j-1]
				if (!is.na(p) & p > 0) logp[[i]] <- logp[[i]] -log(p)
			}
		}
	}
	return(logp)
}
