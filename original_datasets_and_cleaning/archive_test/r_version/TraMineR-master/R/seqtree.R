seqtree <- function(formula, data = NULL, weighted = TRUE, min.size = 0.05,
  max.depth = 5, R = 1000, pval = 0.01, weight.permutation = "replicate",
  seqdist.args = list(method = "LCS", norm = "auto"), diss = NULL, squared = FALSE,
  first = NULL, minSize, maxdepth, seqdist_arg) {

  TraMineR.check.depr.args(alist(min.size = minSize, max.depth = maxdepth, seqdist.args = seqdist_arg))

	##formula.call <- formula
	tterms <- terms(formula)
	seqdata <- eval(formula[[2]], data, parent.frame()) # to force evaluation
	if (!inherits(seqdata, "stslist")) {
		stop("Left hand term in formula should be a stslist object (see seqdef)")
 	}
	if(is.null(diss)){
		seqdist.args$seqdata <- seqdata
		diss <- do.call(seqdist, seqdist.args)
	}
	if(weighted){
		weights <- attr(seqdata, "weights")
	}
	else {
		weights <- NULL
	}
	dissmatrix <- diss
	formula[[2]] <- NULL
	## Model matrix from forumla
	predictor <- as.data.frame(model.frame(formula, data, drop.unused.levels = TRUE, na.action=NULL))
	tree <- DTNdisstree(dissmatrix=dissmatrix, predictor=predictor, terms=tterms,
						weights=weights, min.size=min.size, max.depth=max.depth, R=R,
						pval=pval, object=seqdata, weight.permutation=weight.permutation,
						squared=squared, first=first)
	class(tree) <- c("seqtree", class(tree))
	return(tree)

}


print.seqtree <- function(x, digits = getOption("digits") - 2, medoid=TRUE, ...){
	stslistmedoid <- function(object, index) {
		x <- seqconc(object[index,], void=attr(object,"void"))
		x <- suppressMessages(seqformat(x, from = "STS", to = "SPS", compress = TRUE))
		return(x)
	}
	print.disstree(x, digits = digits, medoid=medoid, medoid.fun=stslistmedoid,...)
}
