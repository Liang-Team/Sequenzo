plot.stslist.meant <- function(x, cpal = NULL, ylab = NULL, yaxis = TRUE,
  xaxis = TRUE, cex.axis = par("cex.axis"), ylim = NULL, bar.labels = NULL,
  cex.barlab = cex.axis, offset.barlab = .1, cex.plot, ...) {

  TraMineR.check.depr.args(alist(cex.axis = cex.plot))

	## Storing the optional graphical parameters in a list
	glist <- list(...)
    parlist <- par()
    glist <- glist[names(glist) %in% names(parlist)]

  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]

  n <- attr(x,"nbseq")
  seql <- length(attr(x,"xtlab"))
  errbar <- attr(x, "se")

  weighted <- attr(x, "weighted")
  if (weighted) {wlab <- "weighted "}
  else {wlab <- NULL}

  if (is.null(ylab))
    ylab <- paste("Mean time (", wlab, "n=",round(n,2),")",sep="")

  if (is.null(ylim))
    ylim <- c(0,seql)

  if (is.null(cpal))
    cpal <- attr(x,"cpal")

  if (sep.ylab) {
      sylab <- ylab
      ylab <- NA
  }

  mt <- as.vector(x[,"Mean"])
  mp <- barplot(mt,
          ## mgp=c(2.5,0.6,0),
          names.arg=if (xaxis) rownames(x) else NULL,
          cex.names=cex.axis,
          cex.axis=cex.axis,
          col=cpal,
          ylim=ylim,
          ylab=ylab,
          axes=FALSE,
          ...)

  ## Plotting the axes
  ## axis(1, at=1:nbstat, labels=ltext, cex.axis=cex.axis)

  if (yaxis){
    plist <- list(side=2, at=round(seq(0, max(ylim), length.out=6),0), cex.axis=cex.axis)
    do.call(axis, args=c(plist,glist))
    #axis(2, at=round(seq(0, max(ylim), length.out=6),0), cex.axis=cex.axis, ...)
  }

  if (errbar){
    se.mt <- x[,"SE"]
    xx <- 1:nrow(x)
    df=attr(x,"nbseq")-1
    if(df >= 1) {
       qt <- qt(.975, df=df)
       tmr.errbar(1.2*xx - .5, mt, mt-qt*se.mt, mt+qt*se.mt, add=TRUE)
    }
    else {
       	warning(paste("Error bars not displayed because df =", df, "too small"))
    }
  }

  if (!is.null(bar.labels)) {
    text(mp, mt + offset.barlab*ylim[2], format(bar.labels), xpd = TRUE, cex = cex.barlab)

  }
    if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)

}
