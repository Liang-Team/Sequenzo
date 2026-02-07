## indexes measuring structure of positive/negative spells (or states)

seqipos <- function(seqdata, dss=NULL, pos.states=NULL, neg.states=NULL, index="share",
    pow=1, w=.5, with.missing=FALSE){

	if (!inherits(seqdata,"stslist"))
		msg.stop("data is NOT a sequence object, see seqdef function to create one")
  if (is.null(pos.states) & is.null(neg.states))
		msg.stop("at least one of pos.states and neg.states should be non null!")
  ind <- c("share","integr","volatility")
  if (index=="integration"){
    msg.warn("index='integration' obsolete, use 'integr' instead")
    index <- "integr"
  }
  if (!index %in% ind)
		msg.stop("index should be one of share, integr, or volatility")
  if (is.null(dss)){
    dss <- index=="share"
  }

  alph <- alphabet(seqdata)
  void <- attr(seqdata,"void")
	nr <- attr(seqdata,"nr")
  if (with.missing)
    alph <- c(alph,nr)

  if (!is.null(pos.states) & !all(pos.states %in% alph)){
    msg.stop("invalid values in pos.states: ",paste(pos.states[!pos.states %in% alph], collapse=",
    "))
  }
  if (!is.null(neg.states) & !all(neg.states %in% alph)){
    msg.stop("invalid values in neg.states: ",paste(neg.states[!neg.states %in% alph], collapse=",
    "))
  }

  if (is.null(pos.states)) pos.states <- alph[!alph %in% neg.states]
  if (is.null(neg.states)) neg.states <- alph[!alph %in% pos.states]

  if (length(pos.states)!=length(unique(pos.states)))
    msg.stop("Multiple occurrences of same state in pos.states")
  if (length(neg.states)!=length(unique(neg.states)))
    msg.stop("Multiple occurrences of same state in neg.states")

  recodes <- list("p"=pos.states, "n"=neg.states)

  if (dss)
    s <- seqdss(seqdata, with.missing = with.missing)
  else
    s <- seqdata

  sbinary <- seqrecode(s, recodes = recodes, otherwise=attr(s,'void'))

  if (index == "share") {
    npos <- rowSums(sbinary=="p")
    nneg <- rowSums(sbinary=="n")
    ret <- npos/(nneg + npos)
  }
  else if (index == "integr"){
    ret <- as.vector(suppressMessages(
      seqintegr(sbinary, state="p", pow=pow, with.missing = with.missing)))
  }
  else if (index == "volatility"){
    ret <- suppressMessages(
      seqivolatility(sbinary, w=w, with.missing = with.missing))
  }
  ret <- as.matrix(ret)
  colnames(ret) <- index
  attr(ret, "sbinary") <- sbinary
  class(ret) <- c(class(ret), "seqipos")

  return(ret)
}

print.seqipos <- function(x, ...){
  names <- dimnames(x)
  attributes(x) <- NULL
  x <- as.matrix(x)
  dimnames(x) <- names
  print(x, ...)
}
