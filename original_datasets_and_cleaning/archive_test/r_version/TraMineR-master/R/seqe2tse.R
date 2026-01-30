seqe2TSE <- function(eseq){
	tse <- .Call(C_tmrseqetotse,  unlist(list(eseq)))
	ll <- levels(eseq)
	tse <- data.frame(id=tse[[1]], timestamp=tse[[2]], event=factor(tse[[3]], levels=1:length(ll), labels=ll))
	return(tse)
}
