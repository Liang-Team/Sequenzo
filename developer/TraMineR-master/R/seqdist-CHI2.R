# Should only be used through seqdist()

CHI2 <- function(seqdata, breaks=NULL, step=1, with.missing=FALSE, norm=TRUE,
          weighted=TRUE, overlap=FALSE, notC=FALSE, euclid=FALSE,
          global.pdotj=NULL, refseq=NULL){
  if(euclid){
		weighted <- FALSE
	}
	if(is.null(breaks)){
		breaks <- list()
		if(step==1){
			for(i in 1:ncol(seqdata)){
				breaks[[i]] <- c(i, i)
			}
		}else if (step==ncol(seqdata)){
			breaks[[1]] <- c(1, ncol(seqdata))
		}else{
			bb <- seq(from=1, to=ncol(seqdata), by=step)
			if(bb[length(bb)]!=(1+ncol(seqdata)-step)){
				msg.warn("With step=",step," last episode is shorter than the others")
			}
			bb <- c(bb, ncol(seqdata)+1)

			bi <- 1
			if(overlap){
				breaks[[bi]] <- c(1, 1+step/2)
				bi <- 2
			}
			for(i in 2:length(bb)){
				breaks[[bi]] <- c(bb[(i-1)], bb[i]-1)
				bi <- bi +1
				if(overlap){
					breaks[[bi]] <- pmin(breaks[[bi-1]]+step/2, ncol(seqdata))
					bi <- bi +1
				}
			}

		}
	}
	if(!is.list(breaks)){
		msg.stop("breaks should be a list of ordered pairs: breaks=list(c(start1, end1), c(start2, end2), ...))")
	}

	labels <- attr(seqdata, "labels")
	cpal <- cpal(seqdata)
	alph <- alphabet(seqdata)
	if(with.missing){
		labels <- c(labels, "missing")
		cpal <- c(cpal, attr(seqdata, "missing.color"))
		alph <- c(alph, attr(seqdata, "nr"))
	}
	nalph <- length(alph)

  if (!is.null(global.pdotj)){
    if(length(global.pdotj)==1) {
      if (global.pdotj[1] != "obs"){
        msg.stop("global.pdotj shlould be either 'obs' or a vector of proportions summing up to 1")
      }else {
        global.pdotj <- seqmeant(seqdata, weighted=weighted, with.missing=with.missing, prop=TRUE)
      }
    } else {
      if (!is.numeric(global.pdotj) || any(global.pdotj<0) || sum(global.pdotj) == 0)
        msg.stop("When a vector, global.pdotj should be non negative with at least one strictly positive element.")
      global.pdotj <- global.pdotj/sum(global.pdotj)
      if (length(global.pdotj) != nalph)
        msg.stop("When a vector, global.pdotj should conform the size of the alphabet including the missing state when applicable")
    }
  }

	weights <- attr(seqdata, "weights")
	if(is.null(weights)|| !weighted){
		weights <- rep(1, nrow(seqdata))
	}
	seqdata <- as.matrix(seqdata)

	dummies <- function(b){
		lastbi <- breaks[[b]][1]
		bi <- breaks[[b]][2]
		bindice <- lastbi:bi
		##blength <- ifelse(norm, length(bindice), 1)
		mat <- matrix(0, nrow=nrow(seqdata), ncol=nalph)
		ndot <- vector("numeric", length=nalph)
		bseq <- seqdata[, bindice]
		myrowSums <- function(x){
			if(!is.null(ncol(x))){
				return(rowSums(x, na.rm=TRUE))
			}else{
				return(x)
			}
		}
		for(i in 1:nalph){
			mat[, i] <- myrowSums(bseq==alph[i]) ##/blength
		}
    ndot <- colSums(weights*mat, na.rm=TRUE) ##GR
    mat <- rbind(mat,ndot) ## GR
    non0rsum <- rowSums(mat, na.rm=TRUE) > 0 ##GR
    mat[non0rsum,] <- mat[non0rsum,]/rowSums(mat[non0rsum,], na.rm=TRUE)## GR
    if (euclid) {
      maxd <- ifelse(norm, 2, 1)
      mat[nrow(mat),] <- rep(maxd, length(ndot))
      mat[nrow(mat),ndot==0] <- 0
    }
    else {
      if(!is.null(global.pdotj)){
        mat[nrow(mat),] <- global.pdotj
        mat[nrow(mat),ndot==0] <- 0
      }
      pdot <- mat[nrow(mat),ndot!=0]
      ## normalize if at least two different states occur in the interval
      ## otherwise distance can only be zero or NA
      if(length(pdot)>1){
        cmin <- c(min(pdot),min(pdot[-which.min(pdot)]))
        maxd <- ifelse(norm, 1/cmin[1] + 1/cmin[2], 1)
        mat[nrow(mat),] <- mat[nrow(mat),] * maxd
      }
    }
    return(mat)
	}
	allmat <- list()
	for(b in 1:length(breaks)){
		allmat[[b]] <- dummies(b)
	}
	allmat <- do.call(cbind, allmat)
  ##print(allmat)
  pdotj <- allmat[nrow(allmat),]
  allmat <- allmat[-nrow(allmat),]
	ndotj <- colSums(allmat) ## pdotj computed in dummies
	cond <- pdotj>0
	allmat <- allmat[, cond]
  #### pdotj defined in dummies for euclid and chi2
	##if(euclid){
	##	pdotj <- rep(1.0, ncol(allmat))
	##} else{
	##  pdotj <- pdotj[cond]
	##}
  pdotj <- pdotj[cond]
	n <- nrow(seqdata)


  ## dealing with refseq as a list (conformity alread checked in seqdist)
  sets <- FALSE
  if (inherits(refseq, "list")) {
      allmat.r <- rbind(allmat[refseq[[1]],],allmat[refseq[[2]],])
      n1 <- length(refseq[[1]])
      n2 <- length(refseq[[2]])
      refseq.id <- c(n1, n1 + n2)
      allmat <- allmat[c(refseq[[1]],refseq[[2]]),]
      sets <- TRUE
  } else if (!is.null(refseq)) {
    refseq.id <- c(refseq,refseq)
    sets <- FALSE
  }

  ##chdist <- function(i, j){
	##	return(sqrt(sum((allmat[i, ]-allmat[j, ])^2/pdotj)))
	##}
	##if(notC){ ##Unused
	##	dd <- numeric(n*(n-1)/2)
	##	for(i in 1:(n-1)){
	##		for(j in (i+1):n){
	##			dd[n*(i-1) - i*(i-1)/2 + j-i] <- chdist(i, j)
	##		}
	##	}
	##}else{
    if (is.null(refseq)) {
  		## SEXP tmrChisq(SEXP ChiTableS, SEXP tdimS, SEXP margeS)
  		dd <- .Call(C_tmrChisq, as.double(allmat), as.integer(dim(allmat)), as.double(pdotj))
    }
    else{
  		dd <- .Call(C_tmrChisqRef, as.double(allmat), as.integer(dim(allmat)), as.double(pdotj), as.integer(refseq.id))
    }
	##}
  if (norm) dd <- dd/sqrt(length(breaks))
  if (sets) {
    dd <- matrix(dd, nrow=n1, ncol=n2, byrow=FALSE, dimnames=list(rownames(seqdata)[refseq[[1]]],rownames(seqdata)[refseq[[2]]]))
  } else if (is.null(refseq)) { ## pairwise distances
    attributes(dd) <- list(Size = n, Labels = rownames(seqdata), Diag = FALSE,
          Upper = FALSE, method = "Chi square sequence", call = match.call(),
          class = "dist")
  } else { ## vector of dist to refseq
    names(dd) <- rownames(seqdata)
  }
	return(dd)
}
