dissvar.grp <- function(diss, group=NULL, ...){

	isdist <- inherits(diss, "dist")
	if (isdist) {
		n <- attr(diss, "Size")
	} else if (is.matrix(diss)) {
		n <- nrow(diss)
	} else {
		stop("diss argument should be a dist object or a dissimilarity matrix")
	}

    grp <- group
    if (is.null(grp)) {
        grp <- rep(1, n)
    }

    if (length(grp) != n){
        stop("length(group) not compatible with size of diss",
            call. = FALSE)
        }

    levg <- levels(grp <- factor(grp))

    v <- vector("double",length(levg))
    ## We need to transform into a matrix to subset
    ## an alternative would be use the subset.dist function from package cba
    if (isdist) diss <- as.matrix(diss)
    for (i in 1:length(levg))
        {
        ig <- which(grp==levg[i])
        v[[i]] <- dissvar(diss[ig,ig], ...)
        }
    names(v) <- levg
    return(v)
}
