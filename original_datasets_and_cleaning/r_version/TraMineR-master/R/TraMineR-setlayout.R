## =====================================
## Setting the layout for TraMineR plots
## =====================================

TraMineR.setlayout <- function(nplot, prows, pcols, with.legend, axes, legend.prop = NA, dom.byrow=NULL) {

	## Backward compatibility
	if (with.legend==TRUE) with.legend <- "auto"

    byrow <- if (is.null(dom.byrow)) TRUE else dom.byrow

	if (is.na(pcols)) pcols <- min(nplot,2)
	if (is.na(prows)) prows <- ceiling(nplot/pcols)

	## Defining initial layout matrix
	pheight <- 1
	pwidth <- 1
	widths <- rep(pwidth/pcols,pcols)
	heights <- rep(pheight/prows,prows)

	layrow <- prows
	laycol <- pcols
	laymat <- matrix(1:(layrow*laycol), nrow=layrow, ncol=laycol, byrow=byrow)

	axisp <- 0

	legpos=NULL
	freecells <- (prows*pcols)-nplot

	## =========================
	## Positioning of the legend
	## =========================
    if (!is.null(dom.byrow)) { ## MD plot
        if (!isFALSE(with.legend)) {
            ## if dom.byrow = TRUE legend  by domain right else bottom
            if (dom.byrow) {
        		if (is.na(legend.prop)) legend.prop <- 0.25
        		laycol <- pcols+1
                layrow <- prows
                widths <- c(rep(1,laycol-1),legend.prop)
                heights <- rep(1,layrow)
                legpos <- "center"
            } else {
        		if (is.na(legend.prop)) legend.prop <- 0.15
                laycol <- pcols
        		layrow <- prows+1
                laymat <- matrix(1:(layrow*laycol), nrow=layrow, ncol=laycol, byrow=byrow)
                widths <- rep(1,laycol-1)
                heights <- c(rep(1,layrow-1),legend.prop)
                legpos <- "bottom"
            }
            laymat <- matrix(1:(layrow*laycol), nrow=layrow, ncol=laycol, byrow=dom.byrow)
        }
    }

	else if (with.legend=="auto") {
		if (freecells==0) {
			if (is.na(legend.prop)) legend.prop <- 0.15
			layrow <- layrow+1

			pheight <- pheight-legend.prop
			heights <- rep(pheight/prows,prows)
			heights <- c(heights,legend.prop)

			widths <- rep(pwidth/laycol,laycol)

			legpos="bottom"

			## Adding one row in the layout matrix for the legend
			laymat <- rbind(laymat, rep(nplot+1,ncol(laymat)))
		}
		else {
			legpos="center"
			heights <- rep(pheight/prows,prows)
			widths <- rep(pwidth/laycol,laycol)
		}
	}
	else if (with.legend=="right") {
		if (is.na(legend.prop)) legend.prop <- 0.25
		laycol <- laycol+1
		pwidth <- pwidth-legend.prop
		legpos="center"
		widths <- rep(pwidth/pcols,pcols)
		widths <- c(widths, legend.prop)
		heights <- rep(pheight/prows,prows)

		## Adding one column in the layout matrix for the legend
		laymat <- cbind(laymat, rep(nplot+1,nrow(laymat)))
	}

	## if (axes %in% c("all","bottom")) axisp <- 1

	## On which plots the axes will appear
    if (is.null(dom.byrow)) {
    	if (axes=="bottom") {
    		for (nc in 1:ncol(laymat))
    			axisp <- c(axisp, max(laymat[laymat[,nc]<=nplot,nc]))
    			#axisp <- c(axisp, max(laymat[1:layrowg,nc]))
    	}
    	else if (axes=="all") axisp <- 1:nplot
    }


	## Returning a list with layout settings
	laylist <- list(laymat=laymat, widths=widths, heights=heights, axisp=axisp, legpos=legpos)

	return(laylist)
}
