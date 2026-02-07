seqivolatility <- function(seqdata, w=.5, with.missing=FALSE, adjust=TRUE){

	if (!inherits(seqdata,"stslist"))
		stop(" [!] data is NOT a sequence object, see seqdef function to create one")
  if (!is.logical(adjust))
		stop(" [!] adjust should be logical")
  if (w>1 | w<0)
		stop(" [!] w should be in the range [0, 1]!")


  alph <- alphabet(seqdata)
  ##void <- attr(seqdata,"void")
	nr <- attr(seqdata,"nr")
  if (with.missing)
    alph <- c(alph,nr)
  alph.size <- length(alph)


  transp <- suppressMessages(
      seqtransn(seqdata, with.missing=with.missing, norm=TRUE))
  sdist <- suppressMessages(
      seqistatd(seqdata, with.missing=with.missing))
  nvisit <- rowSums(sdist>0)
  if (adjust) {
    ret <- ifelse(nvisit - 1 <= 0, 0, (nvisit - 1)/(alph.size -1))
    ret <- w * ret + (1-w) * transp
  } else {
    ret <- w * nvisit/alph.size + (1-w) * transp
  }

  return(as.vector(ret))
}
