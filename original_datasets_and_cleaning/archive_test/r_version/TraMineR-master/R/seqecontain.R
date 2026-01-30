seqecontain <- function(eseq, event.list, unknown.exclude = FALSE, seq,
  eventList, exclude) {

  TraMineR.check.depr.args(alist(eseq = seq, event.list = eventList, unknown.exclude = exclude))

  if(is.subseqelist(eseq))eseq <- eseq$subseq
  if(!is.seqelist(eseq))stop("eseq should be a seqelist. See help on seqecreate.")
  dict<-levels.seqelist(eseq)

  elist<-factor(event.list,levels=dict)
  if(unknown.exclude)excl=as.integer(c(1))
  else excl=as.integer(c(0))
  return(.Call(C_tmrsequencecontainevent, eseq, as.integer(elist), excl))
}
