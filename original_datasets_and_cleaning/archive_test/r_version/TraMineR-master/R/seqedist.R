seqedist <- function(eseq, idcost, vparam, interval=TRUE, norm=TRUE){
    norm <- as.integer(norm)
    interval <- as.integer(interval)
    return(.Call(C_tmrseqedist, eseq, idcost, vparam, norm, interval));
}

seqeage <- function(eseq, event.list) {
	return(.Call(C_tmreventinseq, eseq, as.integer(event.list)))
}
