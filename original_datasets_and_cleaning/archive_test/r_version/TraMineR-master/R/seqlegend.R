## =========================================
## Plotting the legend for a sequence object
## =========================================

seqlegend <- function(seqdata, with.missing = "auto", cpal = NULL,
  missing.color = NULL, ltext = NULL, position = "topleft", cex = 1,
  boxes=TRUE, fontsize, ...) {

  TraMineR.check.depr.args(alist(cex = fontsize))


	if (!inherits(seqdata,"stslist"))
		stop("data is not a sequence object, use seqdef function to create one")

	if (is.null(cpal))
		cpal <- attr(seqdata,"cpal")

	if (is.null(ltext))
		ltext <- attr(seqdata,"labels")

	if (is.null(missing.color))
		missing.color <- attr(seqdata,"missing.color")

	## Adding an entry for missing in the legend
	nr <- attr(seqdata,"nr")

	if ((with.missing=="auto" && any(seqdata==nr)) || with.missing==TRUE) {
		cpal <- c(cpal,missing.color)
		ltext <- c(ltext,"missing")
	## statl <- c(statl,nr)
	## nbstat <- nbstat+1
	}

 	oolist <- list(...)
  if (! "col" %in% names(oolist)) oolist[["col"]] <- cpal
  if (! "x" %in% names(oolist)) oolist[["x"]] <- position
  if (! "legend" %in% names(oolist)) oolist[["legend"]] <- ltext
  oolist <- c(list(cex=cex), oolist)
	plot(0, type= "n", axes=FALSE, xlab="", ylab="")
  if (boxes) {
	  #legend(position, fill=cpal, legend=ltext, cex=cex,...)
    if (! "fill" %in% names(oolist)) oolist[["fill"]] <- cpal
	  res <- do.call(legend,oolist)
  } else {
 	  if (! "lty" %in% names(oolist)) oolist[["lty"]] <- 1
    if (! "lwd" %in% names(oolist)) oolist[["lwd"]] <- 15
    if (! "seg.len" %in% names(oolist)) oolist[["seg.len"]] <- .4
    if (! "x.intersp" %in% names(oolist)) oolist[["x.intersp"]] <- 1.5
	  res <- do.call(legend,oolist)
  }
  invisible(res)
}
