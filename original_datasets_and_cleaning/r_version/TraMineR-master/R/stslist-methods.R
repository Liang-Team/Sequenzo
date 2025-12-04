## ===========================
## Methods for stslist objects
## ===========================

print.stslist <- function(x, format='STS', extended=FALSE, ...) {
	if (format=='STS') {
		if (extended==FALSE) {
			void <- attr(x,"void")
			x <- seqconc(x, void=void)
			print(x, quote=FALSE, ...)
		} else NextMethod("print")
	}
  right <- ifelse(any(as.data.frame(x)[ncol(x)]==attr(x,"nr")), NA, 'DEL')
	if (format=='SPS') {
		x <- seqconc(x, void=attr(x,"void"))

		if (extended==FALSE)
			x <- suppressMessages(seqformat(x, from = "STS", to = "SPS", compress = TRUE, right=right, ...))
		else if (extended==TRUE)
			x <- suppressMessages(seqformat(x, from = "STS", to = "SPS", compress = FALSE, right=right, ...))

		print(x, quote=FALSE)
	}
}

## plot.stslist <- function(x,...) {
##	seqiplot(x)
## }

"[.stslist" <- function(x,i,j,drop=FALSE) {
	## Specialized only for column subscript
	## If one column we keep the original data.frame method
	## Otherwise we copy attributes and update "start" value

  ## For negative j, we first build the new subscript set
  if (!missing(j) && j[1]<0) {
    k <- -j
    j <- 1:ncol(x)
    j <- j[! j %in% k]
  }

    if (!missing(i) && is.logical(i) && any(is.na(i))) {
        #i <- which(i)
        #msg.warn("which applied to subsetting condition because it has NAs")
        warning("NAs in row subsetting condition: use which(condition) to treat them as FALSE!")
    }

    if (!missing(j) && length(j)>1) {
    	## Storing the attributes
    	x.attributes <- attributes(x)

    	## Applying method
        x <- NextMethod("[")

    	## Adapting column names
    	x.attributes$names <- x.attributes$names[j]

    	## Redefining attributes
    	attributes(x) <- x.attributes

        attr(x,"start") <- x.attributes$start-1+j[1]

		if (!missing(i)) {
			attr(x,"row.names") <- attr(x,"row.names")[i]
			attr(x,"weights") <- attr(x,"weights")[i]
		}

		return(x)
	}

	x <- NextMethod("[")

	if (!missing(i))
		attr(x,"weights") <- attr(x,"weights")[i]

	return(x)
}


## "[.stslist" <- function(x,...) {
## 	NextMethod("[")
## }

Math.stslist <- function(...){
 stop("Invalid operation on sequences")
}

rbind.stslist <- function(..., deparse.level = 1) {
  seqlist <- list(...)
  l <- length(seqlist)
  ww <- attr(seqlist[[1]],"weights")
  alph <- alphabet(seqlist[[1]])
  kalph <- 1
  void <-attr(seqlist[[1]],"void")
  nr <-attr(seqlist[[1]],"nr")
  missing.char <-attr(seqlist[[1]],"missing")

  res <- seqlist[[1]]
  n.null <- ifelse(is.null(ww),1,0)
  for (i in 2:l) {
    seqi <- seqlist[[i]]
    weights <- attr(seqi,"weights")
    n.null <- n.null + is.null(weights)
    if (length(alph) < length(alphabet(seqi))) {
      if (!all(alph %in% alphabet(seqi)))
        stop("Alphabet mismatch between stslist objects!")
      alph <- alphabet(seqi)
      kalph <- i
    }
    else {
      if (!all(alphabet(seqi) %in% alph))
        stop("Alphabet mismatch between stslist objects!")
    }
    if (nr != attr(seqi,"nr") || void!= attr(seqi,"void"))
      stop("nr and/or void mismatch between stslist objects!")
    res <- as.matrix(res)
    ## when stslist do not have same number of columns
    ## we adjust with columns of voids
    if (ncol(res) < ncol(seqi)) {
      emptycol <- matrix(void, nrow(res), ncol(seqi)-ncol(res))
      names <- c(names(res),names(seqi)[(ncol(res)+1):ncol(seqi)])
      res <- cbind(res,emptycol)
      names(res) <- names
    }
    else if (ncol(res) > ncol(seqi)) {
      emptycol <- matrix(void, nrow(seqi), ncol(res)-ncol(seqi))
      names <- c(names(seqi),names(res)[(ncol(seqi)+1):ncol(res)])
      seqi <- cbind(seqi,emptycol)
      names(res) <- names
    }
    res <- rbind(as.matrix(res),as.matrix(seqi), deparse.level=deparse.level)
    if (!is.null(weights)) ww <-c(ww,weights)
  }
  if(n.null > 0 & n.null != l)
    stop("!! Cannot rbind stslist objects with and without weights!")

  is.void <- any(res==void)
  res[res == nr] <- missing.char
  res[res == void] <- missing.char

  suppressMessages(
    res <- seqdef(res,
      alphabet=alph,
      weights =ww,
      start   =attr(seqlist[[1]],"start"),
      missing =attr(seqlist[[1]],"missing"),
      nr      =attr(seqlist[[1]],"nr"),
      void    =attr(seqlist[[1]],"void"),
      labels  =attr(seqlist[[kalph]],"labels"),
      xtstep  =attr(seqlist[[1]],"xtstep"),
      cpal    =attr(seqlist[[kalph]],"cpal"),
      tick.last=attr(seqlist[[1]],"tick.last"),
      right   =ifelse(is.void,"DEL",NA)
    )
  )

  return(res)
}

summary.stslist <- function(object,...) {

	alphabet <- alphabet(object)
	nbstates <- length(alphabet)
	cpal <- cpal(object)
	labels <- attr(object,"labels")
	nr <- attr(object,"nr")
	void <- attr(object,"void")
	weights <- attr(object, "weights")
	TraMineR.version <- attr(object, "Version")

	nbseq <- seqdim(object)[1]
	seql <- seqlength(object)
	nuseq <- nrow(unique(object))

	if (!is.null(TraMineR.version)) {
		cat(" [>] sequence object created with TraMineR version",TraMineR.version,"\n")
	}
	cat(" [>]", nbseq, "sequences in the data set,",nuseq, "unique","\n")

	## weights
	if (!is.null(weights) && !all(weights==1)) {
		cat(" [>] sum of weights: ", round(sum(weights),2), " - min/max: ",
			min(weights),"/",max(weights),"\n", sep="")
	}
	cat(" [>] min/max sequence length: ",min(seql),"/",max(seql),"\n", sep="")

	## Alphabet
	cat(" [>] alphabet (state labels): ","\n")
	maxstatedisplay <- 12
	for (i in 1:min(nbstates,maxstatedisplay))
		cat("     ",i, "=", alphabet[i], " (", labels[i], ")","\n", sep="")
	if (nbstates>12) message("      ...")
	cat(" [>] dimensionality of the sequence space:", (nbstates-1)*max(seql),"\n")
	cat(" [>] colors:", paste(1:nbstates,cpal,collapse=" ",sep="="),"\n")

	if (any(object==nr)) {	
		cat(" [>] symbol for missing state:",nr,"\n")
	}

	if (any(object==void)) {	
		cat(" [>] symbol for void element:",void,"\n")
	}
}
