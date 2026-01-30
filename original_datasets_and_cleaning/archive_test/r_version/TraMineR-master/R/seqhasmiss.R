seqhasmiss <- function(seqdata){
  if (!is.stslist(seqdata)){
    msg.stop("seqdata must be a stslist object!" )
  }
  misscode <- c(attr(seqdata,"nr"),attr(seqdata,"void"))
  nrcode <- attr(seqdata,"nr")
  voidcode <- attr(seqdata,"void")
  has.miss <- apply(seqdata,1,function(x) any(x %in% misscode))
  has.nr <- apply(seqdata,1,function(x) any(x %in% nrcode))
  has.void <- apply(seqdata,1,function(x) any(x %in% voidcode))
  cat("There are ",sum(has.miss)," sequences with missings\n", sum(has.nr),"have nr's,", sum(has.void), " have voids")
  return(invisible(list(has.miss=has.miss,has.nr=has.nr,has.void=has.void)))
}
