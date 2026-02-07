## ========================================
## Computes the sequence turbulence measure
## proposed by Elzinga
## ========================================

turb <- function(x) {
		phi <- x[1]
		#n <- x[2]
		## sum.tx <- x[3]
		#mean.tx <- x[3]
		s2.tx <- x[2]


		#s2max <- (n-1) * (1-mean.tx)^2
    s2max <- x[3]
		
		Tux <- log2(phi* ((s2max+1)/(s2.tx+1)))
		return(Tux)
}

## the var function in R gives the unbiased variance with the n-1 denominator
## but we need the real variance here
#realvar <- function(x) {
#	n <- sum(!is.na(x))
#	var <- 1/n*sum((x - mean(x,na.rm=TRUE))^2,na.rm=TRUE)
#	return(var)
#	}
	

seqST <- function(seqdata, norm=FALSE, silent=TRUE, with.missing=FALSE, type=1) {

	if (!inherits(seqdata,"stslist"))
		stop("seqdata is NOT a sequence object, see 'seqdef' function to create one")

## Since v 2.0.13, we use the with.missing argument instead
	##nr <- attr(seqdata,"nr")
	##with.missing=FALSE
	##if (any(seqdata==nr)) {
	##	message(" [!] found missing state in one or more sequences")
	##	message("     [>] adding missing state to the alphabet")
	##	with.missing=TRUE
	##}

	if (!silent) message(" [>] extracting symbols and durations ...")
	spells <- seqdss(seqdata, with.missing=with.missing)
	#dur <- seqdur(seqdata, with.missing=with.missing)

	if (!silent) message(" [>] computing turbulence type ",type," for ",nrow(seqdata)," sequence(s) ...")
	phi <- suppressMessages(seqsubsn(spells, DSS=FALSE, with.missing=with.missing))
    if (any(is.nan(phi))) {
        turb.phi[is.nan(phi)] <- .Machine$double.xmax
        warning("Some phi set as .Machine$double.xmax because exceeding this max allowed value.")
    }
  #s2.tx <- apply(dur, 1, realvar)
  s2.tx <- seqivardur(seqdata, type=type, with.missing=with.missing)
  s2.tx.max <- attr(s2.tx,'vmax')
	#mean.tx <- rowMeans(dur, na.rm=TRUE)
	## sum.tx <- apply(dur, 1, sum, na.rm=TRUE)
	##n <- seqlength(spells, with.missing=with.missing)

	#tmp <- data.frame(phi, n, mean.tx, s2.tx)
	tmp <- data.frame(phi, s2.tx, s2.tx.max)
	Tx <- apply(tmp, 1, turb)
	Tx <- as.matrix(Tx)

    if(norm){
        alph <- alphabet(seqdata)
        if (with.missing) alph <- c(alph,"$%!")

        maxlength <- max(seqlength(seqdata))
        nrep <- ceiling(maxlength/length(alph))

        turb.seq <- suppressWarnings(suppressMessages(seqdef(t(rep(alph,nrep)[1:maxlength]), alphabet=alph)))
        #turb.spells <- seqdss(turb.seq) ##useless if length(alph)>1
        turb.spells <- turb.seq
        #turb.dur <- seqdur(turb.seq)
        if (length(alph)>1)
            turb.phi <- suppressMessages(seqsubsn(turb.spells, DSS=FALSE, with.missing=FALSE))
        else
            turb.phi <- 2
        if (is.nan(turb.phi)) {
            turb.phi <- .Machine$double.xmax
            warning("phi set as .Machine$double.xmax, because it exceeds that value when computing max turbulence")
        }
        #turb.s2 <- apply(turb.dur, 1, realvar)
        turb.s2 <- seqivardur(turb.seq, type=type, with.missing=FALSE)
        turb.s2.max <- attr(turb.s2,'vmax')
        #turb.mean <- rowMeans(turb.dur, na.rm=TRUE)

        #tmp <- data.frame(turb.phi, maxlength, turb.mean, turb.s2)
        tmp <- data.frame(turb.phi, turb.s2, turb.s2.max)

        maxT <- apply(tmp, 1, turb)

        Tx.zero <- which(Tx < 1)

        Tx <- (Tx-1)/as.numeric(maxT-1)

        if (length(Tx.zero)>0) Tx[Tx.zero,] <- 0
    }

	rownames(Tx) <- rownames(seqdata)
	colnames(Tx) <- "Turbulence"

	return(Tx)
}

	
