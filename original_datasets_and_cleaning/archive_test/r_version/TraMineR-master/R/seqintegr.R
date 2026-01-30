seqintegration <- function(...){
  msg.stop("Seqintegration is obsolete, use seqintegr instead!")
}

seqintegr <- function(seqdata, state=NULL, pow=1, with.missing=FALSE){
	if (!inherits(seqdata, "stslist")) {
        stop("[!] seqdata is not a sequence object, see seqdef function to create one")
  }
  alph <- alphabet(seqdata)
  nr <- attr(seqdata, "nr")
  if (with.missing) {
      alph <- c(alph, nr)
  }
  if (length(state)>1)
    msg.stop("When non null, 'state' must be a single state")
  if (!is.null(state)){
    if (!state %in% alph){
        msg.stop("state ", state, " not in the alphabet!")
    }
  }
  nbstat <- ifelse(is.null(state),length(alph),1)
  nbseq <- nrow(seqdata)

  iseqtab <- matrix(nrow = nbseq, ncol = nbstat)
  if (is.null(state))
    colnames(iseqtab) <- alph
  else
    colnames(iseqtab) <- state

  rownames(iseqtab) <- rownames(seqdata)
  ## message(" [>] computing state distribution for ", nbseq, " sequences ...")
	integVector <- (1:ncol(seqdata))^pow
  suminteg <- sum(integVector, na.rm = TRUE)

  ## when with.missing=FALSE we do not account for positions occupied by missings
  miss <- c(nr, attr(seqdata,"void"))
  if(!with.missing)
    suminteg <- suminteg - apply(seqdata, 1, function(x) sum(integVector[x %in% miss], na.rm = TRUE))

  if(is.null(state)){
    for(i in 1:nbstat) {
      iseqtab[, i] <- apply(seqdata, 1, function(x) sum(integVector[x ==  alph[i]], na.rm =
        TRUE))/suminteg
    }
  } else {
        iseqtab[, 1] <- apply(seqdata, 1, function(x) sum(integVector[x ==  state], na.rm = TRUE))/suminteg
  }

  return(iseqtab)
}
