## ========================================
## Return a id of given subsequences
## ========================================

seqeid <- function(eseq, s) {

  TraMineR.check.depr.args(alist(eseq = s))

  tmrsequenceid.internal <- function(eseq) {
    if(is.eseq(eseq)){
      return(.Call(C_tmrsequencegetid, eseq))
    }
    return(NA)
  }
  #message("Event sequence analysis module is still experimental")
  if(is.seqelist(eseq)){
    return(as.integer(sapply(unlist(eseq),tmrsequenceid.internal)))
  }else if(is.eseq(eseq)){
    return(tmrsequenceid.internal(eseq))
  }else{
    stop("eseq should be a seqelist. See help on seqecreate.")
  }
  return(NA)
}
