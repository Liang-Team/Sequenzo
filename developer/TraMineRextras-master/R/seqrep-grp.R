seqrep.grp <- function(seqdata, group = NULL, diss = NULL, ret = "stat", with.missing = FALSE, mdis, ...) {

  ##TraMineR:::checkargs(alist(diss = mdis))
  ## TraMineR.check.depr.args is public version of checkargs GR 19.01.2018
  TraMineR.check.depr.args(alist(diss = mdis))
	if (!inherits(seqdata,"stslist")){
		stop("data is NOT a state sequence object, see seqdef function to create one",
            call. = FALSE)
	}
    if (!(ret %in% c("stat","rep","both"))){
        stop("ret should be one of 'stat', 'rep' or 'both'",
            call. = FALSE)
        }
    grp <- group
    if (is.null(grp)) grp <- rep(1, nrow(seqdata))
    grp <- TraMineR:::group(grp)
    if (length(grp) != nrow(seqdata)){
        stop("length(grp) not equal to number of sequences",
            call. = FALSE)
        }
    if (any(xtabs( ~ grp) < 2))
      stop("Each group must have 2 or more cases. At least one group has only 1.", call.=FALSE)

    levg <- levels(grp <- factor(grp))


		if (is.null(diss)){
      if (! "method" %in% names(list(...))){
			  stop("You must provide a distance matrix or a method to compute it", call.=FALSE)
      } else {
        oolist <- list(...)
        oolist[["seqdata"]] <- seqdata
        oolist[["with.missing"]] <- with.missing
        dlist <- names(formals(seqdist))
        diss <- do.call(seqdist, args = oolist[names(oolist) %in% dlist])
      }
    }
		if (inherits(diss, "dist")) {
      	diss <- as.matrix(diss)
    }

    dmax <- max(diss) ## same dmax for all groups

    q.gr <- gr <- list()
    for (i in 1:length(levg))
        {
        ig <- which(grp==levg[i])
        gr[[i]] <- seqrep(seqdata[ig,], diss=diss[ig,ig], dmax=dmax, with.missing=with.missing, ...)
        q.gr[[i]] <- attr(gr[[i]],"Statistics")
        }
    names(q.gr) <- names(gr) <- levg
    if (ret == "stat") return(q.gr)
    if (ret == "rep")  return(gr)
    if (ret == "both") return(list(gr, q.gr))
}
