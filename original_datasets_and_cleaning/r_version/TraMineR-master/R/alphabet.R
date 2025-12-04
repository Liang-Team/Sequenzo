## ============================================
## Retrieve the alphabet from a sequence object
## ============================================

alphabet <- function(seqdata, with.missing=FALSE) {

	if (inherits(seqdata,c("stslist","PSTf"))){
    statl <- attr(seqdata,"alphabet")
    if (isTRUE(with.missing)) statl <- c(statl, attr(seqdata,"nr"))
  }
  else if (inherits(seqdata,"seqelist")){
    statl <- levels(seqdata)
  }
  else {
		stop("seqdata should be a state sequence object, an event sequence object, or a suffix tree. Use seqdef or seqecreate.")
  }

return(statl)
}
	
