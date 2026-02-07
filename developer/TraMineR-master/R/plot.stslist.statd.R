## ==============================
## PLot of the state distribution
## ==============================

plot.stslist.statd <- function(x, type = "d", cpal = NULL,
    ylab = NULL, yaxis = TRUE,
    xaxis = TRUE, xtlab = NULL, xtstep = NULL,
    tick.last = NULL,
    cex.axis = par("cex.axis"),
    space = 0, xlab = NULL, lwd=3.5, col="blue",
    ylim=NULL, cex.plot, ...) {

  TraMineR.check.depr.args(alist(cex.axis = cex.plot))

  if (!(type %in% c("d", "Ht", "dH"))){
    msg.stop("type can only be 'd', 'Ht', or 'dH'")
  }

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% c("main",names(parlist))]


  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]
  #las <- par("las")
  #if ("las" %in% names(list(...))) las <- list(...)[["las"]]

	n <- attr(x,"nbseq")
	weighted <- attr(x, "weighted")
	if (weighted) {wlab <- "weighted "}
	else {wlab <- NULL}

	if (is.null(xtlab)) {xtlab <- attr(x,"xtlab")}

	if (is.null(xtstep)) {
		if (!is.null(attr(x,"xtstep"))) {xtstep <- attr(x,"xtstep")}
		## For sequence objects created with previous versions
		else {xtstep <- 1}
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}

	seql <- length(xtlab)

    ## for x axis
	x.lab.pos <- NULL
	tpos <- seq(1,seql, xtstep)
    x.pos <- NULL

    if (tick.last & tpos[length(tpos)] < seql) tpos <- c(tpos,seql)

	for (p in tpos) {
		x.lab.pos <- c(x.lab.pos, (p-1)+((p-1)*space)+(0.5+space))
	}
    for (p in 1:seql) {
        x.pos <- c(x.pos, (p-1)+((p-1)*space)+(0.5+space))
    }

    ## y label
    if(is.null(ylab)){
        if (type=="d")
			ylab <- paste("Rel. Freq. (",wlab,"n=",round(n,2),")",sep="")
        else
            ylab <- paste("Entropy (",wlab,"n=",round(n,2),")",sep="")
    }

    if (sep.ylab) {
        sylab <- ylab
        ylab <- NA
    }

    ## y limit for entropy
    y <- x$Entropy
    if (is.null(ylim)) {
        if (isTRUE(attr(x,"norm")))
            ylim <- c(0.025,1)
        else
            ylim <- c(0.025*1.1*max(y),1.1*max(y))
    }


	## State distribution plot
	if (type %in% c("d","dH")) {
		if (is.null(cpal))
			cpal <- attr(x,"cpal")

		#if (is.null(ylab))
		#	ylab <- paste("Freq. (",wlab,"n=",round(n,2),")",sep="")

		bp <- barplot(x$Frequencies,
			space=space,
			axes=FALSE,
			axisnames=FALSE,
			## cex.axis=cex.axis,
			## cex.names=cex.axis,
			col=cpal,
			ylab=ylab,
            xlab=xlab,
			...)

		## Plotting the x axis
 ##		x.lab.pos <- NULL
 ##		tpos <- seq(1,seql, xtstep)
 ##        if (tick.last & tpos[length(tpos)] < seql) tpos <- c(tpos,seql)
 ##
 ##		for (p in tpos) {
 ##			x.lab.pos <- c(x.lab.pos, (p-1)+((p-1)*space)+(0.5+space))
 ##		}

		if (xaxis) {
            plist <- list(side=1, at=x.lab.pos, labels=xtlab[tpos], pos=-0.02, cex.axis=cex.axis)
            do.call(axis, args=c(plist,glist))
			#axis(1, at=x.lab.pos, labels=xtlab[tpos], pos=-0.02, cex.axis=cex.axis, ...)
		}

        if (type == "dH") {
            #if (is.null(ylab.r))
            #    ylab.r <- "Entropy index"
            par(new=TRUE)
            c1 <- 1 - (1-.5*space)/length(y)
            c2 <- .5
            bp <- as.vector(bp)[1:length(y)]
            #plot(x=as.vector(bp)[1:length(y)], y=y, type="n", axes=FALSE,
            plist <- list(x=x.pos, y=y, type="n", axes=FALSE,
                    ylim=ylim, xlab=NA, ylab=NA)
            plist <- c(plist,glist)
            do.call(plot, args=plist)
            #plot(x=x.lab.pos, y=y, type="n", axes=FALSE,
            #        ylim=ylim, xlab=NA, ylab=NA, ...)
            #lines(x=c1*bp + c2, y=y, col=col, lwd=lwd)
            lines(x=c1*x.pos + c2, y=y, col=col, lwd=lwd)
            #axis(4,at=seq(0,max(c(1,max(y))),.2))
            #mtext(ylab.r, line=3, side=4, cex = par("cex"))

        }
	}
	## Entropy index plot
	else if (type=="Ht") {
        #if (is.null(ylab.r))
        #    ylab.r <- paste("Entropy index (",wlab,"n=",round(n,2),")",sep="")
        if (is.null(xlab)) xlab <- NA
        plist <- list(x=y,
			col=col,
			## frame.plot=TRUE,
			type="l",
			lwd=lwd,
			lty="solid",
			axes=FALSE,
			ylim=ylim,
			ylab=ylab,
			xlab=xlab)
        plist <- c(plist,glist)
        do.call(plot, args=plist)

##		plot(y,
##			col=col,
##			## frame.plot=TRUE,
##			type="l",
##			lwd=lwd,
##			lty="solid",
##			axes=FALSE,
##			ylim=ylim,
##			ylab=ylab,
##			xlab=xlab,
##			...)

		## Plotting the x axis
		if (xaxis) {
			#tpos <- seq(1,seql, xtstep)
            plist <- list(side=1, at=tpos, labels=xtlab[tpos],
                        pos=-0.02, cex.axis=cex.axis)
            do.call(axis, args=c(plist,glist))
			#axis(1, at=tpos, labels=xtlab[tpos], pos=-0.02, cex.axis=cex.axis, ...)
		}
	}

	##
    if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)
	if (is.null(yaxis) || yaxis){
        plist <- list(side=2, cex.axis=cex.axis)
        plist <- c(plist,glist)
        do.call(axis, args=plist)

		#axis(2, cex.axis=cex.axis, ...)
    }

  if (type == 'Ht') return(invisible(x$Entropy))
  else if (type == 'd') return(invisible(x$Frequencies))
  else return(invisible(x))
}
