# Authors: Pierre-Alexandre Fonta, Matthias Studer, Gilbert Ritschard (2018)

plot.stslist.surv <- function(x, cpal = NULL, ylab = NULL, xlab = NULL, xaxis = TRUE,
  yaxis = TRUE, xtstep = NULL, tick.last = NULL, #cex.axis = 1,
  ...) {

  sep.ylab <- (isFALSE(yaxis) && (is.null(ylab) || !is.na(ylab)))
  #cex.lab <- par("cex.lab")
  #if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]
  #las <- par("las")
  #if ("las" %in% names(list(...))) las <- list(...)[["las"]]

  if (yaxis) yaxt <- "s" else yaxt <- "n"
  if (is.null(cpal)) cpal <- attr(x, "cpal")
  if (is.null(ylab)) ylab <- "Percentage of unterminated spells"
  if (sep.ylab) {
      sylab <- ylab
      ylab <- NA
  }

	if (is.null(xtstep)) {
		xtstep <- ifelse(!is.null(attr(x, "xtstep")), attr(x, "xtstep"), 1)
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(x, "tick.last")), attr(x, "tick.last"), FALSE)
	}
  if (is.null(xlab)) xlab <- "Time since start of the spell"
  c <- class(x)
  class(x) <- c[c != "stslist.surv"]
  if ("survfit" %in% class(x)) {
    plot(x, col = cpal, xlab = xlab, ylab = ylab, #cex.axis = cex.axis,
      xaxt = "n", yaxt = yaxt, ...)
    if (isTRUE(xaxis)) {
      times <- sort(unique(x$time))
      time.range <- seq(1, max(times))
      tpos <- seq(1, max(times), xtstep)
      if (tick.last & tpos[length(tpos)] < max(times)) tpos <- c(tpos,max(times))
      #axis(1, at = tpos, labels = time.range[tpos], cex.axis = cex.axis, las=las)
      axis(1, at = tpos, labels = time.range[tpos], ...)
    }
   } else { ## x==0 when no cases are selected, plot an empty frame
    plot(0, type = "n", col = cpal, xlab = xlab, ylab = ylab, #cex.axis = cex.axis,
      xaxt = "n", yaxt = "n", ...)
   }
   if (sep.ylab)
    title(ylab=sylab, line=1, ...)

}
