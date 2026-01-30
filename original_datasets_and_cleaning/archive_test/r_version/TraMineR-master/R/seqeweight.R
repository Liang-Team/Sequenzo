## ========================================
## Get and set weight of eseq
## ========================================

seqeweight <- function(eseq, s) {

  TraMineR.check.depr.args(alist(eseq = s))

	seqeweight.internal<-function(eseq){
		if(is.eseq(eseq)) {
			return(.Call(C_tmrsequencegetweight, eseq))
		}
		return(-1)
	}

	if (is.seqelist(eseq)) {
		as.numeric(sapply(unlist(eseq),seqeweight.internal))
	}else if(is.eseq(eseq)) {
		as.numeric(seqeweight.internal(eseq))
	} else {
		stop(" [!] eseq should be a seqelist. See help on seqecreate.")
	}
}

"seqeweight<-" <- function(eseq, s, value) {

  TraMineR.check.depr.args(alist(eseq = s))

	if(!is.seqelist(eseq)) {
		stop(" [!] eseq should be a seqelist. See help on seqecreate.")
	}
	if(length(eseq)!=length(value)) {
		stop(" [!] eseq and weights should be of the same size.")
	}
	.Call(C_tmrsequencesetweight, eseq, as.double(value))
	return(eseq)
}

seqeisweighted <- function(eseq, s) {

  TraMineR.check.depr.args(alist(eseq = s))

	if(!is.seqelist(eseq)) {
		stop(" [!] eseq should be a seqelist. See help on seqecreate.")
	}
	weights <- seqeweight(eseq)
	return(any(weights!=1))
}
