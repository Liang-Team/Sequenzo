seqlength.align <- function(seq.list){

  nsl <- length(seq.list)
  if (!is.list(seq.list))
    stop("seq.list should be a list of two or more stslist objects")


  nr <- nrow(seq.list[[1]])
  lgth <- matrix(NA,nrow=nr,ncol=nsl)
  for (i in 1:nsl) {
    if (!is.stslist(seq.list[[i]]))
        stop("At least one element of seq.list is not a stslist object!")
    if (nrow(seq.list[[i]]) != nr)
        stop("Sequence objects must all have same number of sequences")
    lgth[,i] <- seqlength(seq.list[[i]])
  }
  lgthmin <- apply(lgth,1,min)
  for (i in 1:nsl) {
    for (j in 1:nr) {
      if (lgth[j,i] > lgthmin[j] )
        seq.list[[i]][j,(lgthmin[j]+1):lgth[j,i]] <- attr(seq.list[[i]],"void")
    }
  }


  return(seq.list)
}

