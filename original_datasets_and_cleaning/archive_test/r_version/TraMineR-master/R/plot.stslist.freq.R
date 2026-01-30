## ================================
## PLot of the sequences frequency
## ================================

plot.stslist.freq <- function(x, cpal = NULL, missing.color = NULL, pbarw = TRUE,
  ylab = NULL, yaxis = TRUE, xaxis = TRUE, xtlab = NULL, xtstep = NULL,
  tick.last = NULL, cex.axis = par("cex.axis"), cex.plot, ...) {

  TraMineR.check.depr.args(alist(cex.axis = cex.plot))

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% names(parlist)]

  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]
  space <- NULL
  if ("space" %in% names(list(...))) space <- list(...)[["space"]]

  if (!is.logical(yaxis) && !yaxis %in% c("cum","pct"))
    msg.stop("Bad yaxis value!")

	n <- attr(x,"nbseq")
	weighted <- attr(x, "weighted")
	if (weighted) {wlab <- "weighted "}
	else {wlab <- NULL}
	statl <- attr(x,"alphabet")
	nr <- attr(x,"nr")

	if (is.null(xtlab))
		xtlab <- attr(x,"names")

	if (is.null(xtstep)) {
		if (!is.null(attr(x,"xtstep"))) {xtstep <- attr(x,"xtstep")}
		## For sequence objects created with previous versions
		else {xtstep <- 1}
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}

	seql <- length(xtlab)

	if (is.null(cpal))
		cpal <- attr(x,"cpal")

	## Checking for missing states
	if (any(x==nr)) {
		if (is.null(missing.color)) missing.color <- attr(x,"missing.color")
		cpal <- c(cpal, missing.color)
		statl <- c(statl, nr)
	}

	if (is.null(ylab)) {
		if (yaxis=="pct")
			ylab <- paste("% freq. (",wlab,"n=",n,")",sep="")
        else
			ylab <- paste("Cum. % freq. (",wlab,"n=",round(n,2),")",sep="")
	}

	## Storing the optional parameters in a list
	olist <- list(...)

	tlim <- nrow(x)

	##
	seqbar <- apply(x,1, seqgbar, seql=seql, statl=statl)

	table <- attr(x,"freq")

	if (pbarw==TRUE) barw=table$Percent
	else barw=1

    if (sep.ylab) {
        sylab <- ylab
        ylab <- NA
    }

	## The plot
	barplot(seqbar,col=cpal, width=barw,
		ylab=ylab,
		horiz=TRUE,
		axes=FALSE,
		axisnames=FALSE,
		...)

	## Plotting the x axis
	if (xaxis) {
		tpos <- seq(1, seql, xtstep)
        if (tick.last & tpos[length(tpos)] < seql) tpos <- c(tpos,seql)
        plist <- list(side=1, at=tpos-0.5, labels=xtlab[tpos], cex.axis=cex.axis)
        do.call(axis, args = c(plist,glist))
		#axis(1, at=tpos-0.5, labels=xtlab[tpos], cex.axis=cex.axis, ...)
	}

	## Plotting the y axis
	if (is.null(space))
        space <- 0.2

	if (yaxis==TRUE || is.null(yaxis) || yaxis=="cum") {
		y.lab <- paste(c(0, round(sum(table$Percent),1)),"%",sep="")
		y.tick <- TRUE

		if (!pbarw)
			y.lab.pos <- c(space,(tlim+(tlim*space)))
		else
			y.lab.pos <- c(space*mean(barw),(sum(barw)+(tlim*space*mean(barw))))
	}
	## Percentage frequency of each sequence
	else if (yaxis=="pct") {
		y.lab <- round(table$Percent,1)
		y.tick <- FALSE

		if (!pbarw) {
			y.lab.pos <- 0.7
			for (p in 2:length(y.lab))
				y.lab.pos <- c(y.lab.pos, (p-1)+((p-1)*space)+0.7)
			}
		else {
			y.lab <- y.lab[table$Percent>=0.5]

			y.lab.pos <- (table$Percent[1]/2)+1
			sep <- space*mean(table$Percent)

			for (p in 2:length(y.lab))
				y.lab.pos <- c(y.lab.pos, sum(table$Percent[1:p])+(p*sep)-table$Percent[p]/2)
			}
	}

	if (yaxis==TRUE || yaxis=="cum" || yaxis=="pct"){
        plist <- list(side=2, at=y.lab.pos,
			labels=y.lab,
			tick=y.tick,
			## mgp=c(1.5,1,0),
			##las=1,
			cex.axis=cex.axis)
        do.call(axis, args = c(plist,glist))
##		axis(2, at=y.lab.pos,
##			labels=y.lab,
##			tick=y.tick,
##			## mgp=c(1.5,1,0),
##			##las=1,
##			cex.axis=cex.axis, ...)
    }

    if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)

}
