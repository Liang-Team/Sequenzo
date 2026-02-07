## ==================================
## Returns a vector with the lengths
## of the sequences in seqdata
## ==================================

seqlength <- function(seqdata, with.missing=TRUE) {

	if (!inherits(seqdata,"stslist"))
		stop("data is not a sequence object, use 'seqdef' function to create one")

	sl <- ncol(seqdata)-rowSums( seqdata==attr(seqdata,"void"), na.rm=TRUE )

  if (!with.missing)
	   sl <- sl-rowSums( seqdata==attr(seqdata,"nr"), na.rm=TRUE )

	sl <- as.matrix(sl)
	colnames(sl) <- "Length"
	rownames(sl) <- rownames(seqdata)

	return(sl)
}
