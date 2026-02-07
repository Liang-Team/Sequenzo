## ==============================
## PLOT A REPRESENTATIVE SEQUENCE
## ==============================

plot.stslist.rep <- function(x, cpal = NULL, missing.color = NULL, pbarw = TRUE,
  dmax = NULL, stats = TRUE, ylab = NULL, xaxis = TRUE, xtlab = NULL,
  xtstep = NULL, tick.last = NULL, seq.alt = NULL, info = TRUE,
  cex.with.axis = 1, cex.plot, ...) {

  TraMineR.check.depr.args(alist(cex.with.axis = cex.plot))

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% names(parlist)]


  #las <- par("las")
  #if ("las" %in% names(list(...))) las <- list(...)[["las"]]

	## Extracting attributes
	n <- attr(x,"nbseq")
    nn <- length(attr(x,"Scores"))

    if (!is.null(seq.alt)) {
        if (!is.stslist(seq.alt)) msg.stop("seq.alt should be a state sequence stslist object!")
        if (nrow(seq.alt) != nn) msg.stop("seq.alt should have",nn,"number of sequences")
        attr(seq.alt,"weighted")
        rep.idx <- attr(x,"Index")
        x.alt <- seq.alt[rep.idx,]
        attr(x.alt,"nbseq") <- n
        attr(x.alt,"weighted") <- attr(x,"weighted")
        attr(x.alt,"dmax") <- attr(x,"dmax")
        attr(x.alt,"criterion") <- attr(x,"criterion")
        attr(x.alt,"Statistics") <- attr(x,"Statistics")
        #stats <- FALSE
        x <- x.alt
    }

	if (is.null(xtlab)) {xtlab <- colnames(x)}

	if (is.null(xtstep)) {
		if (!is.null(attr(x,"xtstep"))) {xtstep <- attr(x,"xtstep")}
		## For sequence objects created with previous versions
		else {xtstep <- 1}
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}

	weighted <- attr(x, "weighted")
	if (weighted) {wlab <- "weighted "}
	else {wlab <- NULL}


	seql <- length(xtlab)
	statl <- attr(x,"alphabet")
	nr <- attr(x,"nr")

	criterion <- attr(x,"criterion")
	Statistics <- attr(x,"Statistics")

	if (is.null(dmax)) dmax <- attr(x,"dmax")

	if (is.null(cpal)) cpal <- attr(x,"cpal")

	## Name of the representative sequence
	if (criterion=="dist") ctname <- "centrality"
	else if (criterion=="freq") ctname <- "frequency"
	else if (criterion=="mscore") ctname <- "rep. score"
	else if (criterion=="smode") ctname <- "modal state seq."
	else if (criterion=="prob") ctname <- "probability"
	else if (criterion=="density") ctname <- "density"

	## Adding an entry for missing in the legend
	if (any(x==nr)) {
		if (is.null(missing.color)) missing.color <- attr(x,"missing.color")
		cpal <- c(cpal, missing.color)
		statl <- c(statl, nr)
	}


	## ============================
	## Max distance for axis limits
	## ============================
	nbrep <- nrow(x)

	## Scaling factor for the distance summaries
	dist.scaling <- dmax/seql

	## space between text lines
	vspace <- 0.1

	## Space between seq bars
    space <- 0.2
	if ("space" %in% names(list(...))) space <- list(...)[["space"]]

	##
	if (pbarw)
		barw <- Statistics$"na(%)"[1:nbrep]/100
	else barw=1

	seqbar <- apply(x, 1, seqgbar, seql=seql, statl=statl)

	## The plot
	if (stats) ymax <- 2.5
	else ymax <- 1.3

	if (is.null(ylab)) {
		ylab <- paste(nbrep, " representative(s) (", wlab, "n=", round(n,2),")",sep="")
	}

    sline = -1
	if (is.na(ylab) || ylab=="") {
        if(isFALSE(stats)){
            xmin <- 0
        } else {
            xmin <- -1.5
        }
    }
    else  xmin <- -2

    barplot(seqbar,col=cpal, width=barw,
		ylab=NA,
		xlim=c(xmin,seql),
		ylim=c(0,ymax),
		horiz=TRUE,
		axes=FALSE,
		axisnames=FALSE,
		...)

    if (!is.na(ylab)){
        cex.lab <- par("cex.lab")
        if ("cex.lab" %in% names(list(...)))
            cex.lab <- list(...)[["cex.lab"]]
        title(ylab=ylab, line=0+stats, cex.lab=cex.lab)
    }

	## Time axis for the sequence
	if (xaxis) {
		tpos <- seq(1,seql, xtstep)
        if (tick.last & tpos[length(tpos)] < seql) tpos <- c(tpos,seql)

        plist <- list(side=1, at=tpos-0.5, labels=xtlab[tpos],
			pos=-0.04)
        do.call(axis, args=c(plist,glist))

##		axis(1, at=tpos-0.5, labels=xtlab[tpos],
##			pos=-0.04,
##			## mgp=c(.5,.5,0),
##			##cex.axis=cex.with.axis,
##            ...)
	}

	## y (percents) axis

	## y.lab.pos <- c(space*mean(barw),(1+(nrow(x)*space*mean(barw))))

	## axis(2, at=y.lab.pos,
	##	labels=c("0%", "100%"),
	##	las=1,
	##	cex.axis=cex.with.axis)

	## Frequency of the representative sequence
    if (info) {
    	nbprox <- sum(Statistics$nb[1:nbrep])
    	ctfreq <- round((nbprox/n)*100,1)
	    text(seql/2, 1.3,
		  paste("Criterion=",ctname,", coverage=", ctfreq ,"%", sep=""),
		  cex=cex.with.axis)
    }

	## ==========
	## Statistics
	## ==========
	if (stats) {
		## Symbols
		repcol <- brewer.pal(10,"Paired")
		repsymb <- c(21:25,15:19)

		## Start position
		y.sym.pos <- 1.8

		## Symbols for the representative sequences
		spval <- space/nbrep
		y.lab.pos <- (barw[1]/2)+spval
		lines(-1, y.lab.pos, type="b", pch=repsymb[1], lwd=3,
			col=repcol[1], cex=cex.with.axis+barw[1])
		for (p in 2:nbrep) {
			y.lab.pos <- sum(barw[1:p-1])+(p*spval)+(barw[p]/2)
			lines(-1, y.lab.pos, type="b", pch=repsymb[p], lwd=3,
				col=repcol[p], cex=cex.with.axis+barw[p])
		}

		## Distance to representative seq.
		dist.rep.pos <- y.sym.pos

		for (i in 1:nbrep) {
			lines(Statistics$MD[i]/dist.scaling, dist.rep.pos,
				type="b", pch=repsymb[i], lwd=3, col=repcol[i], cex=1+barw[i])
		}

		legend.B <- paste("(B) Mean dist. to representative seq.", sep="")

		## Distance to center
		y.sym.pos <- y.sym.pos + 2*vspace
		dist.center.pos <- y.sym.pos

		for (i in 1:nbrep) {
			lines(Statistics$V[i]/dist.scaling,
				y.sym.pos,
				type="b", pch=repsymb[i], lwd=3, col=repcol[i], cex=cex.with.axis+barw[i])
		}

		legend.A <- paste("(A) Discrepancy (mean dist. to center)",sep="")

		## Axis for the boxplot of distances to the representative sequences
		dypos <- 1.7

		nbdec <- if (dmax>=4) 0 else 1

        plist <- list(side=1, at=seq(0,seql,seql/4),
			labels=round(seq(0,dmax,dmax/4),nbdec),
			pos=dypos, mgp=c(.5,.5,0))
        do.call(axis, args=c(plist,glist))

##		axis(1, at=seq(0,seql,seql/4),
##			labels=round(seq(0,dmax,dmax/4),nbdec),
##			pos=dypos, mgp=c(.5,.5,0),
##			...)

		axis(2, at=c(dist.rep.pos,dist.center.pos),
            pos = 0,
			labels=c("B","A"),
			las=2,
			cex.axis=cex.with.axis)

		legend(seql/2, ymax, legend=c(legend.A, legend.B),
			xjust=0.5,
			yjust=0.7,
			## title="Distance",
			box.lty=0,
			cex=cex.with.axis)
	}
}
