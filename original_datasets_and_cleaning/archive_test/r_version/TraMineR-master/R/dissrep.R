## ====================================
## Extracting a set of representatives
## ====================================

dissrep <- function(diss, criterion = "density", score = NULL, decreasing = TRUE,
  coverage = 0.25, nrep = NULL, pradius = 0.10, dmax = NULL, weights = NULL,
  trep, tsim) {

  TraMineR.check.depr.args(alist(coverage = trep, pradius = tsim))

	if (inherits(diss, "dist")) {
        diss <- dist2matrix(diss)
	}

    diss <- as.matrix(diss) #gr
    diss0 <- if (all(diss==0)) TRUE else FALSE
	nbobj <- nrow(diss)

	if (is.null(weights)) { weights <- rep(1, nbobj) }
	else if (length(weights)!= nbobj) {
		stop(" [!] number of provided weigths must equal the number of objects")
	}

	weights.sum <- sum(weights)

	message(" [>] number of objects (sum of weights): ", round(weights.sum,2))

	## Max theoretical distance
	if (is.null(dmax)) { dmax <- max(diss) }

	if (pradius<0 || pradius>1) { stop("pradius must be between 0 and 1", call.=FALSE) }
	pradius <- dmax*pradius

	message(" [>] max. distance: ", round(dmax,2))
	message(" [>] neighborhood radius: ", round(pradius,2))

	## =====================
	## Neighbourhood density
	## =====================
	if (criterion=="density") {

		neighbours <- diss<pradius
		score <- as.vector(neighbours %*% weights)
		decreasing <- TRUE
	}
	## ===========
	## Frequencies
	## ===========
	else if (criterion=="freq") {

		neighbours <- diss==0
		score <- as.vector(neighbours %*% weights)

		decreasing <- TRUE
	}
	## ============
	## Min distance
	## ============
	else if (criterion=="dist") {
		## Sum of distances for all distinct sequences
		score <- as.vector(diss %*% weights)
		decreasing <- FALSE
	}
	## ======
	## random
	## ======
	else if (criterion=="random") {
		##
		score <- sample(1:nbobj, nbobj)
		decreasing <- FALSE
	}
	## =======
	## unknown
	## =======
	else if (is.null(score))
		stop("Unknown criterion / no score provided")

	## ===========================
	## Sorting candidates by score
	## ===========================
	if (length(score)!=nrow(as.matrix(diss)))
		stop("Score must be a vector of length equal to",nbobj)

	score.sort <- order(score, decreasing=decreasing)
	rep.dist <- diss[score.sort, score.sort, drop=FALSE]

	## ==========================
	## Selecting representatives
	## ==========================
	idx <- 0
	idxrep <- NULL

	## Coverage fixed
	if (is.null(nrep) && coverage>0) {
		pctrep <- 0

		while (pctrep<coverage && idx < nbobj) {
			## Searching for next non-redundant sequence in candidate list
			idx <- idx+1
			if (idx==1 || all(rep.dist[idx, idxrep]>pradius)) {
				idxrep <- c(idxrep, idx)
				tempm <- as.matrix(rep.dist[, idxrep, drop=FALSE])
				nbnear <- sum((rowSums(tempm<pradius)>0)*weights[score.sort])
				pctrep <- nbnear/weights.sum
			}
		}

		nbkeep <- length(idxrep)
		message(" [>] ", nbkeep, " representative(s) selected, coverage=",
			round(pctrep,2)*100,"% (threshold=",round(coverage,2)*100,"%)")
	}
	## Number of desired representative fixed
	else {
		repcount <- 0

		while (repcount<nrep && idx<nbobj) {
			## Searching for next non-redundant sequence in candidate list
			idx <- idx+1
			if (idx==1 || all(rep.dist[idx, idxrep]>pradius)) {
				## message(" [>] Adding object with index", score.sort[idx], " to representative set")
				idxrep <- c(idxrep, idx)
				repcount <- repcount+1
			}
		}

		nbkeep <- length(idxrep)
		message(" [>] ", nbkeep, " representative(s) selected")
	}

	## On force avec as.matrix car sinon il y a une erreur
	## si 1 seule colonne
	dist.repseq <- as.matrix(diss[,score.sort[idxrep], drop=FALSE])

	## ================
	## Quality measures
	## ================

	## Keeping distance to the nearest representative sequence only
	dc.tot <- disscenter(diss, weights=weights)

	if (nbkeep>1) {
		tied <- 0
		minidx <- apply(dist.repseq,1, which.min)

		for (i in 1:nbobj) {
			dist.repseq[i,-minidx[i]] <- NA
		}

		## message(" [>] ",tied," observations equaly distant from two or more representatives")

		na <- colSums((!is.na(dist.repseq))*weights)
    		SD <- colSums(dist.repseq*weights, na.rm=TRUE)
		MD <- SD/na

		## Number of similar sequences attributed to each representative
		nb <- colSums((dist.repseq < pradius)*weights, na.rm=TRUE)

		## DC: Sum of distances to global center
    		## V: Inertia
		DC <- matrix(nrow=nbkeep,ncol=1)
    		V <- matrix(nrow=nbkeep,ncol=1)
		for (i in 1:nbkeep) {
      			sel <- !is.na(dist.repseq[,i])
			DC[i] <- sum(dc.tot[sel]*weights[sel])
			tmp <- as.matrix(diss[sel, sel])
			V[i] <- mean(disscenter(tmp, weights=weights[sel]))
		}

	} else {
		na <- sum((!is.na(dist.repseq))*weights)
		SD <- sum(dist.repseq*weights, na.rm=TRUE)
    		MD <- SD/sum(weights)
		nb <- sum((dist.repseq < pradius)*weights, na.rm=TRUE)
		DC <- sum(dc.tot*weights)
		V <- DC/sum(weights)
        minidx <- rep(1,nrow(dist.repseq))
	}
    if (diss0) nb<-sum(weights, na.rm=TRUE)


    ## List of ids of representatives in original data
    dist.to.rep <- apply(dist.repseq,1, min, na.rm=TRUE)
    idx.rep <- list()
    for  (i in 1:nbkeep){
        idx.rep[[i]] <- which(minidx==i & dist.to.rep==0)
    }

	quality <- (sum(dc.tot*weights)-sum(dist.repseq*weights, na.rm=TRUE))/sum(dc.tot*weights)

	## Overall stats
	na <- c(na,sum(na))
	nb <- c(nb,sum(nb))
  	SD <- c(SD, sum(dist.repseq*weights, na.rm=TRUE))
	MD <- c(MD, sum(dist.repseq*weights, na.rm=TRUE)/sum(weights))
	DC <- c(DC, sum(dc.tot*weights))
	V <- c(V, sum(dc.tot*weights)/sum(weights))

	pcta <- (na/weights.sum)*100
	pctb <- (nb/weights.sum)*100
	Q <- (DC-SD)/DC*100

	stats <- data.frame(na, pcta, nb, pctb, SD, MD, DC, V, Q)
	colnames(stats) <- c("na", "na(%)", "nb", "nb(%)", "SD", "MD", "DC", "V", "Q")
	rownames(stats) <- c(paste("r",1:nbkeep,sep=""), "Total")

    ## list of cases represented by each representatives
    #lidx <- apply(dist.repseq,2,function(x) {which(!is.na(x))})

	## ============
	## Final object
	## ============
	res <- score.sort[idxrep]
	class(res) <- c("diss.rep", class(res))

	attr(res, "n") <- weights.sum
	attr(res, "criterion") <- criterion
	attr(res, "dmax") <- dmax
	attr(res, "Scores") <- score
	attr(res, "Distances") <- dist.repseq
    attr(res, "Rep.group") <- minidx
    attr(res, "idx.rep") <- idx.rep
	attr(res, "Statistics") <- stats
	attr(res, "Quality") <- quality

	return(res)
}
