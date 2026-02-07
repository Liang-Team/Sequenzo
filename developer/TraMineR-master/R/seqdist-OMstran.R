# Should only be used through seqdist()

OMstran <- function(seqdata, indel, sm, full.matrix, transindel, otto, previous,
  add.column, with.missing, weighted, refseq, norm) {

  alph <- alphabet(seqdata, with.missing=with.missing)
  void <- attr(seqdata,"void")
	dimnames(sm) <- list(alph, alph)
	if(length(indel)==1){
		indel <- rep(indel, length(alph))
	}
	names(indel) <- alph
	if(transindel=="prob"){
		tr <- seqtrate(seqdata, with.missing=with.missing, weighted=weighted)
		dimnames(tr) <- list(alph, alph)
	}
  sl <- seqlength(seqdata)
	seqdata <- as.matrix(seqdata)
	maxcol <- ncol(seqdata)
	newseqdata <-matrix("", nrow=nrow(seqdata), ncol=maxcol)
	sep <- "@@@@TraMineRSep@@@@"

	mypastefunc <- function(i){
		minmax <- function(i){
			return(max(1, min(i, maxcol)))
		}
    tostate <- seqdata[,minmax(i+1)]
    # voids can only be at end of sequences
    # when to state is void we set it as the previous state when add.column
    if (add.column){
      tostate[tostate==void] <- seqdata[tostate==void,minmax(i)]
    }
		ret <- paste( seqdata[, minmax(i)], tostate, sep=sep)
    if (add.column){
      # set transition as NA when from state is void
      ret[seqdata[,minmax(i)]==void] <- NA
    }else{
      # set transition as NA when to state is void
      ret[tostate==void] <- NA
    }
    if(previous){
      ret.na <- is.na(ret)
			ret <- paste(seqdata[, minmax(i-1)], ret, sep=sep)
      ret[ret.na] <- NA
      ## ret[seqdata[,minmax(i-1)]==void] <- NA  ## cannot happen for !ret.na
		}
		return(ret)
	}

	for (i in 1:(maxcol)) {
		newseqdata[, i] <- mypastefunc(i)
	}
	if(!add.column){
		newseqdata <- newseqdata[, -maxcol]
		if(previous){
			newseqdata <- newseqdata[, -1]
		}
	}
	## transition to void states have been set as NA  (to/from nr will be considered as a distinct transition)

  newalph <- unique(as.character(newseqdata))
  newalph <- newalph[!is.na(newalph)] ## deleting NAs
	alphabet_size <- length(newalph)
	suppressMessages(newseqdata <- seqdef(newseqdata, cpal=rep("blue", alphabet_size),
                                      left='DEL', gaps='DEL', right='DEL',
                                      id = rownames(seqdata)) )
	transweight <- 1 - otto
	if(previous){
		transweight <- transweight/2
	}
	indelrate <- max(indel)/max(sm)
	transweight <- transweight* indelrate
	indel <- indelrate*indel/max(indel)

	sm <- sm/max(sm)
	## =========================================
	## Building the new substitution cost matrix
	## =========================================

	## Build substitution matrix and new alphabet
	alphabet <- alphabet(newseqdata)
	alphabet_size <- length(alphabet)
	msg("Creating", alphabet_size, "distinct transition states")
	## Recomputing the subsitution matrix
	indels <- numeric(alphabet_size)
	names(indels) <- alphabet
	newsm <- matrix(0, nrow=alphabet_size, ncol=alphabet_size)
	stateindel <- indel* otto
	## indel costs change according to the previous parameters
	if(!previous){
		for(i in 1:alphabet_size){
			statesi <- strsplit(alphabet[i], sep)[[1]]
			indels[i] <- stateindel[statesi[1]]
			if(statesi[1]!= statesi[2]){
				if(transindel=="constant"){
					 rawtransindel <- 1
				}else if(transindel=="prob"){
					rawtransindel <- (1-tr[statesi[1], statesi[2]])
				}else if(transindel=="subcost"){
					rawtransindel <- (sm[statesi[1], statesi[2]])
				}
				indels[i] <- indels[i] +  transweight*rawtransindel
			}
		}

	}else{
		for(i in 1:alphabet_size){
			statesi <- strsplit(alphabet[i], sep)[[1]]
			rawtransindel <- 0
			if(transindel=="constant"){
				 rawtransindel <- (statesi[1]!= statesi[2])+(statesi[2]!= statesi[3])
			}else if(transindel=="prob"){
				rawtransindel <- (2-tr[statesi[1], statesi[2]]-tr[statesi[2], statesi[3]])
			}else if(transindel=="subcost"){
				rawtransindel <- (sm[statesi[1], statesi[2]]+sm[statesi[2], statesi[3]])
			}
			indels[i] <- stateindel[statesi[2]] +  transweight*rawtransindel
		}


	}
	## Just fix the state we are comparing according to 'previous'
	compare_state <- ifelse(previous, 2, 1)
	for (i in 1:(alphabet_size-1)) {
		statesi <- strsplit(alphabet[i], sep)[[1]]
		for (j in (i+1):alphabet_size) {
			statesj <- strsplit(alphabet[j], sep)[[1]]
			cost <- sm[statesi[compare_state], statesj[compare_state]]
			if(transindel %in% c("constant", "prob", "subcost")){
				cost <- otto*cost + (indels[alphabet[i]] + indels[alphabet[j]] -stateindel[statesi[compare_state]] -stateindel[statesj[compare_state]])
			}
			newsm[i, j] <- cost
			newsm[j, i] <- cost
		}
	}
	suppressMessages(return(seqdist(newseqdata, method = "OM", indel = indels, sm = newsm, full.matrix = full.matrix,
      weighted = weighted, refseq = refseq, norm = norm)))
}
