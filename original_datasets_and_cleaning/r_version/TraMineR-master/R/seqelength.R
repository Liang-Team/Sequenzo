## ========================================
## Get and set length of eseq
## ========================================

seqelength <- function(eseq, s) {

  TraMineR.check.depr.args(alist(eseq = s))

	seqelength.internal<-function(eseq){
		if(is.eseq(eseq)){
			return(.Call(C_tmrsequencegetlength, eseq))
		}
		return(-1)
	}
	if(is.seqelist(eseq)){
		as.numeric(sapply(unlist(eseq),seqelength.internal))
	}else if(is.eseq(eseq)){
		as.numeric(seqelength.internal(eseq))
	}else{
		stop("eseq should be a seqelist. See help on seqecreate.")
	}
}

"seqelength<-" <- function(eseq, s, value){

  TraMineR.check.depr.args(alist(eseq = s))

	if(!is.seqelist(eseq)) {
		stop("eseq should be a seqelist. See help on seqecreate.")
	}
	if(length(eseq)!=length(value)) {
		stop("eseq and len should be of the same size.")
	}
	.Call(C_tmrsequencesetlength, eseq, as.double(value))
	return(eseq)
}
