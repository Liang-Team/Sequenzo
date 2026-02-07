## ====================================================
## Generic function for plotting state sequence objects
## ====================================================

seqplot <- function(seqdata, group = NULL, type = "i", main = "auto",
  cpal = NULL, missing.color = NULL,
  ylab = NULL, yaxis = "all",
  xaxis = "all", xtlab = NULL, cex.axis = 1,
  with.legend = "auto", ltext = NULL, cex.legend = 1,
  use.layout = (!is.null(group) | with.legend != FALSE), legend.prop = NA,
  rows = NA, cols = NA, title, cex.plot, withlegend, axes, ...) {

  TraMineR.check.depr.args(alist(main = title, cex.axis = cex.plot, with.legend = withlegend, xaxis=axes))

	if (!inherits(seqdata,"stslist"))
		stop(call.=FALSE, "seqplot: data is not a state sequence object, use seqdef function to create one")

	## Storing original optional arguments list
	oolist <- list(...)

  	if ("sortv" %in% names(oolist)) {sortv <- oolist[["sortv"]]}
  	leg.ncol <- if ("ncol" %in% names(oolist)) { oolist[["ncol"]] } else { NULL }
    oolist <- oolist[names(oolist) != "ncol"]
    if ("tlim" %in% names(oolist)) {
      oolist[["idxs"]] <- oolist[["tlim"]]
      msg.warn("'tlim' deprecated, use 'idxs' instead!")
      oolist <- oolist[names(oolist) != "tlim"]
    }
    barlab <- if ("bar.labels" %in% names(oolist)) { as.matrix(oolist[["bar.labels"]]) } else { NULL }

    diss <- NULL
  	if ("diss" %in% names(oolist)) {
      diss <- oolist[["diss"]]
    }
  	else if ("dist.matrix" %in% names(oolist)) {
      diss <- oolist[["dist.matrix"]]
      oolist[["diss"]] <- diss
    } #  dist.matrix is deprecated

    ## Stuff for rf plot
    use.rf.layout <- FALSE
    if ("which.plot" %in% names(oolist)){
        #msg.warn("which.plot ignored, because not allowed as seqplot argument!")
        #if (length(oolist) > 0) oolist <- oolist[[names(oolist) != "which.plot"]]
        #else oolist <-  list()
        if (oolist[["which.plot"]] == "both"){
            if (!is.null(group))
                msg.stop('which.plot="both" applies only when group=NULL')
            use.layout <- use.rf.layout <- TRUE
        }
    }

    if (is.logical(xaxis)){
        xaxis <- ifelse (xaxis, "all", FALSE)
    } else {
        if (!xaxis %in% c("all","bottom"))
            msg.stop('If not logical, xaxis should be one of "all" or "bottom"')
    }
    axes <- xaxis
    fyaxis <- "cum"
    if (is.logical(yaxis)){
        yaxis <- ifelse(yaxis, "all", FALSE)
    } else if (type == "f"){
        if (!yaxis %in% c("all","left","left.pct","pct","cum","left.cum"))
            msg.stop('If not logical, yaxis should be one of "all","left","pct","cum","left.pct","left.cum"')
        if (yaxis == "cum") {
            yaxis <- "all"
        } else if (yaxis == "left.cum") {
            yaxis <- "left"
        } else if (yaxis == "pct") {
            yaxis <- "all"
            fyaxis <- "pct"
        } else if (yaxis == "left.pct") {
            yaxis <- "left"
            fyaxis <- "pct"
        }

    } else {
        if (!yaxis %in% c("all","left"))
            msg.stop('If not logical, yaxis should be one of "all" or "left"')
    }
    yaxes <- yaxis

  if (type == "pc") { # modification of Reto Bürgin 16.08.2012
    oolist <- append(oolist, list(group = group, rows = rows, cols = cols))
    group <- NULL
  }

  if (type == "r") { # stuff moved here by GR 17.01.2018
    ## For type="r" each group should have at least 2 cases
    grp <- group
    if (is.null(grp)) grp <- rep(1,nrow(seqdata))
    if (any(xtabs( ~ group(grp)) < 2))
      stop("For type = 'r', each group must have 2 or more cases. At least one group has only 1.", call.=FALSE)

		if (is.null(diss)) {## (! "diss" %in% names(oolist)  && ! "dist.matrix" %in% names(oolist))){
      if (! "method" %in% names(oolist)){
			  stop("For type = 'r', you must provide a distance matrix or a method to compute it", call.=FALSE)
      } else {
        #msg("seqplot calls seqdist")
        oolist[["seqdata"]] <- seqdata
        dlist <- names(formals(seqdist))
        diss <- do.call(seqdist, args = oolist[names(oolist) %in% dlist])
        oolist[["diss"]] <- diss
        # removing seqdist args
        oolist <- oolist[!(names(oolist) %in% dlist[dlist != "weighted"])]
      }
    }
		#else { ## GR: should also be checked on the seqdist outcome
			if (inherits(diss, "dist")) {
      				diss <- dist2matrix(diss)
			}
    #}
    ## Setting unique Max theoretical distance for all groups
		if (!"dmax" %in% names(oolist)) {
			dmax <- max(diss)
			oolist <- c(oolist,list(dmax=dmax))
		}
  }

  if (type=="rf"){
    if (is.null(diss))
        msg.stop("'diss' required for rf plots")
	with.missing <- TRUE

	if ("sortv" %in% names(oolist))
        sortv <- oolist[["sortv"]]
    else
        sortv <- "mds"
    if (length(sortv)==1 && sortv=="mds"){
        weighted <- TRUE
        if ("weighted" %in% names(oolist)) weighted <- oolist[["weighted"]]
        if (weighted) {
           if ("weights" %in% names(oolist))
             weights <- oolist[["weights"]]
           else
             weights <- attr(seqdata,"weights")
           if (is.null(weights)) {
             weighted <- FALSE
           }
        }

        mdspow <- 1
        if ("squared" %in% names(oolist)) mdspow <- 2^oolist[["squared"]]
        if (weighted)
            sortv <- wcmdscale(diss^mdspow, k = 1, w=weights)
        else
            sortv <- cmdscale(diss^mdspow, k = 1)
    }
  }

	## ==============================
	## Preparing if group is not null
	## ==============================

	if (!is.null(group)) {
          group <- group(group)

          ## Check length
          if (length(group)!=nrow(seqdata))
            stop(call.=FALSE, "group must contain one value for each row in the sequence object")

          nplot <- length(levels(group))
          gindex <- vector("list",nplot)

          if (length(ylab) <= 1) ## length(NULL) is 0
            ylab <- rep(ylab, nplot)
          else if (length(ylab) != nplot)
            msg.stop("If a vector, ylab must have one value per group level!")

          if (type=="mt" & !is.null(barlab)){
            if (!(ncol(barlab) %in% c(1,nplot)) )
            stop(call.=FALSE, "When a matrix, bar.labels should have one column per group")
          }

          for (s in 1:nplot)
            gindex[[s]] <- which(group==levels(group)[s])

          ## Title of each plot
          #if (!is.null(main))
          #  main <- paste(main,"-",levels(group))
          #else
          #  main <- levels(group)


          if (!is.null(main)) {
              if (main[1] == "auto")
                main <- levels(group) ## will be NULL if group is NULL
              else if (length(main)==1)
                main <- paste(main,"-",levels(group))
          }

	} else { # single group
          nplot <- 1
          gindex <- vector("list",1)
          gindex[[1]] <- 1:nrow(seqdata)
          if (!is.null(main) && main[1] == "auto" && type!="pc") main <- NULL
	}

	## ===================
	## Defining the layout
	## ===================
	if (type=="Ht" | type =="pc") { with.legend=FALSE }

    ## Issue below fixed by using xaxis instead of axes ## gr 22.11.23
    ## IF xaxis argument is provided
	## it interferes with axes argument
## 	if ("xaxis" %in% names(oolist)) {
## 		tmpxaxis <- oolist[["xaxis"]]
## 		if (tmpxaxis==TRUE) {axes="all"}
## 		else if (tmpxaxis==FALSE) {axes=FALSE}
## 		oolist <- oolist[!names(oolist) %in% "xaxis"]
## 	}


	if (use.layout | !is.null(group) ) {
		## Saving graphical parameters
		savepar <- par(no.readonly = TRUE)


        if (use.rf.layout) {
		  lout <- TraMineR.setlayout(2, 1, 2, with.legend, axes, legend.prop)
          lout$widths[1] <- 1.3*lout$widths[1]
	  	  layout(lout$laymat, heights=lout$heights, widths=lout$widths)
        }
        else {
		  lout <- TraMineR.setlayout(nplot, rows, cols, with.legend, axes, legend.prop)
	  	  layout(lout$laymat, heights=lout$heights, widths=lout$widths)
        }
		## Should axis be plotted or not ?
		#xaxis <- 1:nplot==lout$axisp

        xaxis <- rep(FALSE,nplot)
        xaxis[lout$axisp] <- TRUE

        yaxis <- as.list(rep(FALSE,nplot))
        if (!isFALSE(yaxes)){
            if (yaxes == "left"){
                #print(lout$laymat[,1])
                if (type == "f")
                    yaxis[lout$laymat[,1]] <- fyaxis
                else
                    yaxis[lout$laymat[,1]] <- TRUE
            }
            else if (yaxes == "all") {
                if (type == "f")
                    yaxis <- as.list(rep(fyaxis,nplot))
                else
                    yaxis <- as.list(rep(TRUE,nplot))
            }
        }
        #print(yaxis)

		legpos <- lout$legpos
	}
	else {
		if (isFALSE(axes)) {xaxis <- FALSE}
		else {xaxis <- TRUE}
        if (isFALSE(yaxes)) {yaxis <- FALSE}
        else yaxis <- TRUE
		legpos <- NULL
	}

	## ========
	## Plotting
	## ========
    if ("col.entr" %in% names(oolist)){
        col.entr <- oolist[["col.entr"]]
        oolist <- c(list(col=col.entr), oolist[!"col.entr" %in% names(oolist)])
    }
	for (np in 1:nplot) {
		## Storing ... arguments in a list
		olist <- oolist

		plist <- list(main=main[np], cpal=cpal, missing.color=missing.color,
			ylab=ylab[np], yaxis=yaxis[[np]], xaxis=xaxis[np],
			xtlab=xtlab, cex.axis=cex.axis)

		## Selecting sub sample for x
		## according to 'group'
		subdata <- seqdata[gindex[[np]],]
        if ("weights" %in% names(olist) & !is.null(weights)){
            olist[["weights"]] <- weights[np]
        }
		## State distribution plot or Entropy index
		if (type=="d" || type=="Ht" || type=="dH") {
			f <- seqstatd
			plist <- c(list(type=type), plist)

			plist <- plist[!names(plist) %in% "missing.color"]

			## Removing the 'cpal' argument which is not used
			## in Entropy index plots
			if (type=="Ht") {plist <- plist[!names(plist) %in% "cpal"]}
		}
		## Sequence frequency plot
		else if (type=="f") {
			with.missing <- TRUE
			f <- seqtab
		}
		## Sequence index plot
		else if (type=="i" || type=="I") {
			f <- function(seqdata) {return(seqdata)}
			with.missing <- TRUE

			## Selecting sub sample for sort variable
			## according to 'group'
			if ("sortv" %in% names(olist)) {
				if (!length(sortv)==1) {
					olist[["sortv"]] <- sortv[gindex[[np]]]
				}
			}

			if (type=="I") {
				if (!"idxs" %in% names(olist)) {olist <- c(olist, list(idxs=0))}
				if (!"space" %in% names(olist)) {olist <- c(olist, list(space=0))}
				if (!"border" %in% names(olist)) {olist <- c(olist, list(border=NA))}
			}
		}
		## Sequence relative frequency plot
		else if (type=="rf") {
			f <- seqrf
			## Selecting sub sample for sort variable
			## according to 'group'
			if (!is.null(sortv) & !length(sortv)==1) {
				olist[["sortv"]] <- sortv[gindex[[np]]]
			}

			if (!"space" %in% names(olist)) {olist <- c(olist, list(space=0))}
			if (!"border" %in% names(olist)) {olist <- c(olist, list(border=NA))}
			## Selecting distances according to group
			olist[["diss"]] <- diss[gindex[[np]],gindex[[np]]]
            plist[["skipar"]] <- TRUE

            if (use.rf.layout) olist[["which.plot"]] <- "medoids"
		}
		## Mean times
		else if (type=="mt") {
          f <- seqmeant
          if (!is.null(barlab)) {
            if (ncol(barlab)==1)
              olist[["bar.labels"]] <- as.vector(barlab)
            else
              olist[["bar.labels"]] <- as.vector(barlab[,np])
          }
        }
		## Modal states
		else if (type=="ms") {
			f <- seqmodst
		}
		## Representative sequence
		else if (type=="r") {
			f <- seqrep
			with.missing <- TRUE

			## Removing unused arguments
			plist <- plist[!names(plist) %in% "yaxis"]

			## Selecting distances according to group
			# fixed for deprecated dist.matrix

			olist[["diss"]] <- diss[gindex[[np]],gindex[[np]]]

    # FIXME dist.matrix is deprecated
    olist <- olist[names(olist) != "dist.matrix"]

    }
    else if (type == "pc") { # modification of Reto Bürgin 16.08.2012

                        plist[["main"]] <- list(main)
                        olist <- c(olist, plist)
                        olist[["plot"]] <- FALSE
                        f <- seqpcplot
                        olist <- olist[names(olist) %in% names(formals(f))]
                        plist <- list()
                      }
		else
			stop("Unknown 'type' argument.")

		## Calling appropriate function and plotting
		flist <- names(formals(f))

		if ("with.missing" %in% names(olist)) {
			with.missing <- olist[["with.missing"]]
		} else if ("with.missing" %in% flist) {
			with.missing <- formals(f)$with.missing
		}

		## Xlim when plotting individual sequences
		if (type %in% c("i", "I", "f")) {
			if (!"xlim" %in% names(olist)) {
				olist <- c(olist, list(xlim=c(0, ncol(seqdata))))
			}
		}

		match.args <- names(olist) %in% flist
		fargs <- olist[match.args]
		fargs <- c(list(seqdata=subdata), fargs)
    #msg(paste("do.call(",f, fargs,")"))
		res <- do.call(f, args=fargs)

		olist <- olist[!match.args]
    ## suppress non plot arguments if necessary
    olist <- olist[!names(olist) %in% c("with.missing")]
    if (!(type %in% c("i","I","rf"))) olist <- olist[!(names(olist) %in% c("sortv","weighted"))]
    if (type != "r") olist <- olist[!(names(olist) %in% c("dmax","stats"))]

		plist <- c(list(x=res), plist, olist)
		do.call(plot, args=plist)

        if (use.rf.layout){
            plist[["which.plot"]] <- "diss.to.med"
            plist[["ylab"]] <- NA
            plist[["yaxis"]] <- FALSE
            do.call(plot, args=plist)
        }
	}

	## Plotting the legend
	if (!is.null(legpos)) {
		## Extracting some sequence characteristics
		nr <- attr(seqdata,"nr")

		if (is.null(ltext)) ltext <- attr(seqdata,"labels")

		if (is.null(missing.color)) missing.color <- attr(seqdata,"missing.color")

		if (is.null(cpal)) cpal <- attr(seqdata,"cpal")

		## no longer needed, we pass oolist to legend
        ##density <- if ("density" %in% names(oolist)) { oolist[["density"]] } else { NULL }
		##angle <- if ("angle" %in% names(oolist)) { oolist[["angle"]] } else { NULL }

		## Adding an entry for missing in the legend
		if (with.missing & any(seqdata==nr)) {
			cpal <- c(cpal,missing.color)
			ltext <- c(ltext,"missing")
		## statl <- c(statl,nr)
		## nbstat <- nbstat+1
		}

        legargs <- names(formals(legend))
        largs <- oolist[names(oolist) %in% legargs]
        largs <- largs[!names(largs) %in% c("cex","col")]
        largs <- c(list(legpos, ltext, cpal, cex=cex.legend, leg.ncol=leg.ncol),largs)

		#TraMineR.legend(legpos, ltext, cpal, cex=cex.legend, density=density, angle=angle, leg.ncol=leg.ncol)
		do.call(TraMineR.legend, largs)
	}

	## Restoring graphical parameters
	if (use.layout | !is.null(group)) {par(savepar)}
}
