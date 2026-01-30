## ====================
## Plotting the legend
## ====================

TraMineR.legend <- function(pos, text, colors, cex=1, leg.ncol = NULL, ... ) {

	nbstat <- length(text)

	## Computing some parameters for the legend's plotting '

  if (is.null(leg.ncol)) {
	   if (pos=="bottom") {
  		if (nbstat > 6)
  			nbcol <- 3
  		else
  			nbcol <- 2

  		leg.ncol <- ceiling(nbstat/nbcol)
	   }
	   else
		  leg.ncol <- 1
   }

	## leg.inset <- -0.2 + ((2-leg.ncol)*0.025)

	## Setting graphical parameters while saving them in savepar
	savepar <- par(mar = c(1, 1, 0.5, 1) + 0.1, xpd=FALSE)

	## Restoring graphical parameters
	on.exit(par(savepar))

	plot(0, type = "n", axes = FALSE, xlab = "", ylab = "")
		## legend(position, fill = cpal, legend = ltext, cex = fontsize)

    oolist <- list(...)
    oolist <- oolist[!names(oolist) %in% c("x","y","legend","ncol","fill","border", "lty", "lwd")]

    legargs <- list(x=pos, legend=text, fill=colors, ncol=leg.ncol, cex=cex, border="black")
    legargs <- c(legargs,oolist)

    do.call(legend, legargs)


## 	legend(pos,
## 		## inset=c(0,leg.inset),
## 		legend=text,
## 		fill=colors,
## 		ncol=leg.ncol,
## 		##bty="o",
## 		cex=cex,
## 		...)

}
