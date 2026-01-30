## =============================
## Plot the modal state sequence
## =============================

plot.stslist.modst <- function(x, cpal = NULL, ylab = NULL, yaxis = TRUE,
  xaxis = TRUE, xtlab = NULL, xtstep = NULL, tick.last = NULL,
  info = TRUE, cex.axis = par("cex.axis"), las = 1, cex.plot, ...) {

  TraMineR.check.depr.args(alist(cex.axis = cex.plot))

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% names(parlist)]

  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]

	seql <- ncol(x)
	statl <- attr(x,"alphabet")
	n <- attr(x, "nbseq")
	nr <- attr(x,"nr")

	if (is.null(cpal)) {cpal <- attr(x,"cpal")}

	## Adding an entry for missing in the legend
	if (any(x==nr)) {
		missing.color <- attr(x,"missing.color")
		cpal <- c(cpal, missing.color)
		statl <- c(statl, nr)
	}
	nbstat <- length(statl)

	weighted <- attr(x, "weighted")
	if (weighted) {wlab <- "weighted "}
	else {wlab <- NULL}

	if (is.null(xtlab)) {xtlab <- colnames(x)}
	if (is.null(xtstep)) {
		if (!is.null(attr(x,"xtstep"))) {xtstep <- attr(x,"xtstep")}
		## For sequence objects created with previous versions
		else {xtstep <- 1}
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}

	if (is.null(ylab)) ylab <- paste("State freq. (",wlab,"n=",round(n,2),")",sep="")

	## ============================
	## Max distance for axis limits
	## ============================
	mod <- attr(x, "Frequencies")
	prof.freq <- matrix(0, nrow=nbstat, ncol=seql)

	## Preparing the matrix for plot
	for (i in 1:seql) {
		smax <- which(statl==x[,i])
		prof.freq[smax,i] <- mod[i]
	}

	## Frequency of the representative sequence
	nbrep <- attr(x,"Occurrences")
	ctfreq <- round((nbrep/n)*100,1)
	txt <- paste("Modal state sequence (",nbrep," occurrences, freq=", ctfreq ,"%)", sep="")

    if (sep.ylab) {
        sylab <- ylab
        ylab <- NA
    }


	barplot(prof.freq,
		space=0,
		## mgp=c(2.5,0.6,0),
		cex.names=cex.axis,
		ylim=c(0,1.2),
		col=cpal,
		## main=title,
		axisnames=FALSE,
		ylab=ylab,
		axes=FALSE,
		...)

	if(info) text(seql/2, 1.1, txt, cex=cex.axis)

	## Plotting the x axis
	if (xaxis) {
		tpos <- seq(1,seql, xtstep)
        if (tick.last & tpos[length(tpos)] < seql)
            tpos <- c(tpos,seql)
        plist <- list(side=1,
            at=tpos-0.5,
            labels=xtlab[tpos],
            pos=-0.02,
            las=las)
        do.call(axis, args=c(plist,glist))
##    		axis(1, at=tpos-0.5, labels=xtlab[tpos], pos=-0.02,
##    		# mgp=c(3,0.5,0),
##    		cex.axis=cex.axis, las=las, ...)
	}

	## Axis for the state frequencies
	if (yaxis){
        plist <- list(side=2,
            at=seq(0,1.0,0.25),
            labels=c("0","0.25",".5","0.75","1"),
			cex.axis=cex.axis, las=las)
        do.call(axis, args=c(plist,glist))
##		axis(2, at=seq(0,1.0,0.25), labels=c("0","0.25",".5","0.75","1"),
##			cex.axis=cex.axis, las=las, ...)
    }
if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)


}
