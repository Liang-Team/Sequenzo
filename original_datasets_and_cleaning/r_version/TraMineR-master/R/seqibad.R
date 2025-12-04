seqibad <- function(seqdata, pow=1, with.missing=FALSE, ...){
	if (!inherits(seqdata, "stslist")) {
        stop("[!] seqdata is not a sequence object, see seqdef function to create one")
  }
  alph <- alphabet(seqdata, with.missing=with.missing)
  lalph <- length(alph)

  stprec <- suppressMessages(seqprecstart(seqdata, with.missing=with.missing, ...))
  integr <- suppressMessages(seqintegr(seqdata, with.missing=with.missing, pow=pow))

  bad <- stprec[1] * integr[,1]

  if (lalph > 1) {
    for (i in 2:lalph){
      bad <- bad + stprec[i] * integr[,i]
    }
  }

  return(bad)

}
