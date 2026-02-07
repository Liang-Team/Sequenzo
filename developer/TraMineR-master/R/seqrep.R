## =============================================
## Representative sequence of a set of sequences
## =============================================

seqrep <- function(seqdata, criterion = "density", score = NULL,
  decreasing = TRUE, coverage = 0.25, nrep = NULL, pradius = 0.10, dmax = NULL,
  diss = NULL, weighted = TRUE, trep, tsim, dist.matrix, ...) {

  TraMineR.check.depr.args(alist(coverage = trep, pradius = tsim, diss = dist.matrix))

	if (!inherits(seqdata,"stslist"))
		stop("data is not a sequence object, see seqdef function to create one", call.=FALSE)

	slength <- ncol(seqdata)
	statelist <- alphabet(seqdata)

	weights <- attr(seqdata, "weights")

	if (!weighted || is.null(weights)) { weights <- rep(1, nrow(seqdata)) }
	if (all(weights==1)) { weighted <- FALSE }

	## Distance matrix
	if (missing(diss) || is.null(diss))
		diss <- seqdist(seqdata, ...)


	if (is.null(score)) {
		## ====================
		## Max representativity
		## ====================
		if (criterion=="mscore") {
			message(" [!] criterion still under development")

			## State distribution
			freq <- seqstatd(seqdata)$Frequencies

			score <- apply(seqdata,1, TraMineR.mscore, slength, statelist, freq)
			decreasing <- TRUE
		}
		## ===============
		## Max probability
		## ===============
		else if (criterion=="prob") {

			score <- seqlogp(seqdata)
			decreasing <- FALSE
		}
	}

	## Getting the representatives
	rep <- dissrep(diss, criterion=criterion, score=score,
		decreasing=decreasing, coverage=coverage, nrep=nrep, pradius=pradius, dmax=dmax, weights=weights)

	## Occurrence of the representative sequence
	nds <- nrow(unique(seqdata))
	message(" [>] ", nds, " distinct sequence(s)")

	## ============
	## Final object
	## ============
	res <- seqdata[rep,]
	rownames(res) <- paste("[",1:nrow(res),"]", sep="")
	class(res) <- c("stslist.rep", class(res))

	attr(res, "nbseq") <- attr(rep, "n")
	attr(res, "criterion") <- criterion
	attr(res, "dmax") <- attr(rep,"dmax")
	attr(res, "Index") <- as.vector(rep)
	attr(res, "Scores") <- attr(rep,"Scores")
	attr(res, "Distances") <- attr(rep,"Distances")
	attr(res, "Rep.group") <- attr(rep,"Rep.group")
    attr(res, "idx.rep") <- attr(rep, "idx.rep")
	attr(res, "Statistics") <- attr(rep,"Statistics")
	attr(res, "Quality") <- attr(rep,"Quality")
	attr(res, "weighted") <- weighted

	return(res)
}
