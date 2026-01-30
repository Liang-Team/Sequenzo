## =============================
## PLot of STS sequence objects
## =============================

plot.stslist <- function(x, idxs = NULL, weighted = TRUE, sortv = NULL,
  cpal = NULL, missing.color = NULL, ylab = NULL, yaxis = TRUE, xaxis = TRUE,
  ytlab = NULL, las = par("las"), xtlab = NULL, xtstep = NULL, tick.last = NULL,
  cex.axis = par("cex.axis"), tlim, cex.plot, ylas, ...) {

  TraMineR.check.depr.args(alist(idxs = tlim, cex.axis = cex.plot, las = ylas))

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% names(parlist)]

  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]
  space <- NULL
  if ("space" %in% names(list(...))) space <- list(...)[["space"]]
  #las <- par("las")
  #if ("las" %in% names(list(...))) las <- list(...)[["las"]]

	n <- nrow(x)
	seql <- ncol(x)
	statl <- alphabet(x)
	nr <- attr(x,"nr")

	if (is.null(xtlab))
		xtlab <- colnames(x)

	if (is.null(xtstep)) {
		if (!is.null(attr(x,"xtstep"))) {xtstep <- attr(x,"xtstep")}
		## For sequence objects created with previous versions
		else {xtstep <- 1}
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}

	## Range
	if (is.null(idxs)) {
		if (n>=10) idxs <- 1:10
		else idxs=1:n
	}
	else if (idxs[1]==0)
			idxs <- 1:n
	else if (max(idxs) > n)
			idxs <- 1:n

	## Sorting
	if (!is.null(sortv)) {
        if (length(sortv)==1 && sortv %in% c("from.start", "from.end")) {
        		end <- if (sortv=="from.end") { max(seqlength(x)) } else { 1 }
        		beg <- if (sortv=="from.end") { 1 } else { max(seqlength(x)) }

        	sortv <- do.call(order, unname(as.data.frame(x))[,end:beg])
        	x <- x[sortv,]
        } else if (length(sortv)!=n) {
            stop(call.=FALSE, "sortv must contain one value for each row in the sequence object ",
                "or be either 'from.start' or 'from.end'")
        } else {
        	if (is.factor(sortv)) { sortv <- as.integer(sortv) }
        	x <- x[order(sortv),]
        }

        sortlab <- paste(", sorted")

	} else { sortlab <- NULL }

	##
	if (is.null(cpal))
		cpal <- attr(x,"cpal")

	## Adding an entry for missing in the legend
	if (any(x==nr)) {
		if (is.null(missing.color)) missing.color <- attr(x,"missing.color")
		cpal <- c(cpal, missing.color)
		statl <- c(statl, nr)
	}

	ssamp <- x[idxs,]
	seqbar <- apply(ssamp, 1, seqgbar, statl=statl, seql=seql)

	## WEIGHTS
	## Weights
	weights <- attr(x, "weights")

	if (!weighted || is.null(weights)) {
		weights <- rep(1.0, nrow(x))
	}
	## Also takes into account that in unweighted sequence objects created with
	## older TraMineR versions the weights attribute is a vector of 1
	## instead of NULL
	if (all(weights==1))
		weighted <- FALSE

	if (weighted) {wlab <- "weighted "}
	else {wlab <- NULL}

	if (is.null(ylab)) {
		ylab <- paste(length(idxs)," seq. ", "(", wlab,"n=", round(sum(weights),2),")",
			sortlab, sep="")
	}

    if (sep.ylab) {
        sylab <- ylab
        ylab <- NA
    }

	## The PLot
    barplot(seqbar,col=cpal, width=weights,
		ylab=ylab,
		horiz=TRUE,
		yaxt="n",
		axes=FALSE,
		#las=1,
		...
	)

	## Plotting the x axis
	if (xaxis) {
		tpos <- seq(from=1, to=seql, by=xtstep)
        if (tick.last & tpos[length(tpos)] < seql) tpos <- c(tpos,seql)
        plist <- list(side=1, at=tpos-0.5, labels=xtlab[tpos],
		              cex.axis=cex.axis, las=las)
        plist <- c(plist,glist)
        do.call(axis, args=plist)

##		axis(1, at=tpos-0.5, labels=xtlab[tpos],
##		# mgp=c(3,0.5,0),
##		cex.axis=cex.axis, las=las, ...)
	}

	## Plotting the y axis
	if (is.null(yaxis) || yaxis) {
		if (is.null(space))
            sp <- 0.2
		else
            sp <- space

		idxmax <- length(idxs)

		if (!weighted) {
			y.lab.pos <- sp+0.5

			if (idxmax>1) {
				for (p in 2:idxmax) {
					y.lab.pos <- c(y.lab.pos, (p-1)+((p-1)*sp)+(0.5+sp))
				}
			}
		}
		else {
			y.lab.pos <- (weights[1]/2)+sp
			sep <- sp*mean(weights)

			if (idxmax>1) {
				for (p in 2:idxmax)
					y.lab.pos <- c(y.lab.pos, sum(weights[1:p])+(p*sep)-weights[p]/2)
			}
		}

		if (is.null(ytlab)) {ytlab <- idxs}
		else if (length(ytlab)==1) {
            if(ytlab=="id")
                {ytlab <- rownames(x)[idxs]}
            else if (length(idxs)>1)
                stop(paste("Bad ytlab value",ytlab))
        }
        else if (length(ytlab)!=length(idxs))
                stop("Length of ytlab does not match number of sequences!")

        plist <- list(side=2, at=y.lab.pos, mgp=c(1.5,0.5,0), labels=ytlab, las=las, tick=FALSE, cex.axis=cex.axis)
        plist <- c(plist,glist)
        do.call(axis, args=plist)
		##axis(2, at=y.lab.pos, mgp=c(1.5,0.5,0), labels=ytlab, las=las, tick=FALSE, cex.axis=cex.axis, ...)
	}

    if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)

}
