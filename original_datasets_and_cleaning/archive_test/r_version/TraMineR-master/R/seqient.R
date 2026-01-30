## =======================
## Within Sequence Entropy
## =======================

seqient <- function(seqdata, norm=TRUE, base=exp(1), with.missing=FALSE, silent=TRUE) {

	if (!inherits(seqdata,"stslist"))
		stop("data is NOT a sequence object, see seqdef function to create one")

	statl <- attr(seqdata,"alphabet")

	if (with.missing) {
		statl <- c(statl, attr(seqdata,"nr"))
	}

	if (!silent) message(" [>] computing entropy for ",nrow(seqdata)," sequences ...")

	iseqtab <- suppressMessages(seqistatd(seqdata, with.missing=with.missing))

	ient <- apply(iseqtab,1,entropy, base=base)
	ient <- as.matrix(ient)
	if (norm==TRUE) {
		emax <- log(length(statl))
		ient <- ient/emax
		}

	colnames(ient) <- "Entropy"
	rownames(ient) <- rownames(seqdata)

	return(ient)

	}
