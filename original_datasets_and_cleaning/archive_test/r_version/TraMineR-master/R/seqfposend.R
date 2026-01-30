seqfposend <- function(seqdata,state, with.missing=FALSE, lead=0, from.seq.start=TRUE){
  if (!inherits(seqdata,"stslist")) {
  	stop("data is not a sequence object, use 'seqdef' function to create one")
  }
  s.dss <- seqdss(seqdata, with.missing=with.missing)
  pos <- seqfpos(s.dss,state)
  s.dur <- seqdur(seqdata, with.missing)
  if (from.seq.start) {
    ## cumulated duration
    s.dur <- t(apply(s.dur,1,cumsum))
  }
  tl <- vector(length=nrow(s.dur))
  for (i in 1:nrow(s.dur)){
    tl[i] <- ifelse(is.na(pos[i]),0,s.dur[i,pos[i]] + lead)
  }
  return(tl)
}
