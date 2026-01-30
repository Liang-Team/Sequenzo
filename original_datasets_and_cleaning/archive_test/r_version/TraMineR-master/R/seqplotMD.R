## ====================================================
## Generic function for plotting multidimensional state sequences
## ====================================================

seqplotMD <- function(channels, group = NULL, type = "i", main = NULL,
  cpal.dom = NULL, missing.color = NULL,
  ylab = NULL, yaxis = "all",
  xaxis = "all", xtlab = NULL, stats = "all", cex.axis = 1,
  with.legend = "auto", ltext.dom = NULL,
  cex.legend = 1, legend.prop = ifelse(dom.byrow,.25,.15),
  dom.byrow = FALSE, dom.crit = 0, dnames=names(channels), ...) {

  #TraMineR.check.depr.args(alist(main = title, cex.axis = cex.plot, with.legend = withlegend))

  ndom <- length(channels)
  if (!is.list(channels) || ndom < 2)
    msg.stop("channels should be a list of at least two paired state sequence objects")
  ncase <- nrow(channels[[1]])

  for (i in 1:ndom){
	if (!inherits(channels[[i]],"stslist"))
		msg.stop("at least one element of channels is not a state sequence object!")
    if (nrow(channels[[i]]) != ncase){
		msg.stop("domains with different number of sequences!")
    }
  }
  if (is.null(dnames)) dnames <- paste0('Dom',1:ndom)
  if (is.null(cpal.dom))
    cpal.dom <- vector("list",ndom)
  else if (!is.list(cpal.dom) || length(cpal.dom) != ndom)
    msg.stop("cpal.dom should be a list with a color palette for each domain")

  if (is.null(ltext.dom))
    ltext.dom <- vector("list",ndom)
  else if (!is.list(ltext.dom) || length(ltext.dom) != ndom)
    msg.stop("ltext.dom should be a list with a color palette for each domain")

  if (is.logical(xaxis)){
      xaxis <- ifelse (xaxis, "all", FALSE)
  } else {
      if (!xaxis %in% c("all","bottom"))
          msg.stop('If not logical, xaxis should be one of "all" or "bottom"')
  }
  axes <- xaxis
  if (is.logical(yaxis)){
      yaxis <- ifelse(yaxis, "all", FALSE)
  } else if (type=="f") {
      if (!yaxis %in% c("all","left","left.pct","pct","cum","left.cum"))
          msg.stop('If not logical, yaxis should be one of "all","left","pct","cum","left.pct","left.cum"')
      else if (yaxis=="cum")
        yaxis <- "all"
      else if (yaxis=="left.cum")
        yaxis <- "left"
  } else {
      if (!yaxis %in% c("all","left"))
          msg.stop('If not logical, yaxis should be one of "all" and "left"')
  }
  yaxes <- yaxis
  if (is.logical(stats)){
      stats <- ifelse (stats, "all", FALSE)
  } else {
      if (!stats %in% c("all","first"))
          msg.stop('If not logical, stats should be one of "all" or "first"')
  }
  which.stats <- stats

  if (!is.logical(dom.byrow))
      msg.stop("dom.byrow must be logical!")


  ## Storing original optional arguments list
  oolist <- list(...)

  sortv <- if ("sortv" %in% names(oolist))  oolist[["sortv"]] else NULL
  if (type=="rf" && !"sortv" %in% names(oolist)) sortv <- "mds" ## default for rf plot
  leg.ncol <- if ("ncol" %in% names(oolist)) { oolist[["ncol"]] } else { NULL }
  oolist <- oolist[names(oolist) != "ncol"]

  #if ("tlim" %in% names(oolist)) {
  #  oolist[["idxs"]] <- oolist[["tlim"]]
  #  msg.warn("'tlim' deprecated, use 'idxs' instead!")
  #  oolist <- oolist[names(oolist) != "tlim"]
  #}
  ##MD We need bar.labels by domain x group, i.e. a list of bar.labels matrices
  if ("bar.labels" %in% names(oolist)) {
   if (!is.list(oolist[["bar.labels"]]) || length(oolist[["bar.labels"]]) != ndom)
      msg.stop("bar.labels must be a list with bar labels for each domain")
   barlab <- list()
   for (i in 1:ndom) {
      barlab[[i]] <- as.matrix(oolist[["bar.labels"]][[i]])
   }
  } else { barlab <- NULL }


  diss <- NULL
    if ("diss" %in% names(oolist)) { ## should be a MD dist matrix
        diss <- oolist[["diss"]]
    } else if ("dist.matrix" %in% names(oolist)) {
        diss <- oolist[["dist.matrix"]]
        oolist[["diss"]] <- diss
    } #  dist.matrix is deprecated

  ## Stuff for rf plot
  use.rf.layout <- FALSE
  if ("which.plot" %in% names(oolist)){
      #msg.warn("which.plot ignored, because not allowed as seqplot argument!")
      #if (length(oolist) > 0) oolist <- oolist[[names(oolist) != "which.plot"]]
      #else oolist <-  list()
      ##MD which.plot="both" not applicable for MD sequences
      if (oolist[["which.plot"]] == "both"){
          #if (!is.null(group))
              msg.stop('which.plot="both" not applicable for MD plots')
          #use.rf.layout <- TRUE
      }
  }

  ##MD pc plot may need a special handling for MD data!!
  ## for now, we disable the "pc" type
  #if (type == "pc") { # modification of Reto Bürgin 16.08.2012
    #msg.stop('type="pc" not implemented for MD plots')
    #oolist <- append(oolist, list(group = group, rows = NA, cols = NA))
    #group <- NULL
  #}

  if (type == "r") { # stuff moved here by GR 17.01.2018
    ## For type="r" each group should have at least 2 cases
    grp <- group
    if (is.null(grp)) grp <- rep(1,nrow(channels[[1]]))
    if (any(xtabs( ~ group(grp)) < 2))
      msg.stop("For type = 'r', each group must have 2 or more cases. At least one group has only 1.")

	if (is.null(diss)) {## (! "diss" %in% names(oolist)  && ! "dist.matrix" %in% names(oolist))){
      #if (! "method" %in% names(oolist)){
			  msg.stop("For type = 'r', you must provide a distance matrix")
      #}
      ##MD we need dissimilarities between MD sequences, which cannot be obtained here by calling seqdist
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



	## ==============================
	## Preparing if group is not null
	## ==============================

	if (!is.null(group)) {
          group <- group(group)

          ## Check length
          if (length(group)!=nrow(channels[[1]]))
            msg.stop("group must contain one value for each row in the sequence object")

          nplot <- length(levels(group))
          gindex <- vector("list",nplot)

          for (s in 1:nplot)
            gindex[[s]] <- which(group==levels(group)[s])

          if (length(ylab) <= 1) ## length(NULL) is 0
            ylab <- rep(ylab, nplot)
          else if (length(ylab) != nplot)
            msg.stop("if a vector, ylab must have one value per group level!")


	} else { # single group
          nplot <- 1
          gindex <- vector("list",1)
          gindex[[1]] <- 1:nrow(channels[[1]])
	}
    ## Title of each plot
    if (!is.null(main))
    main <- paste(main,"- ")


    ## =====================
    ## preparing domains
	## ===================


    for (d in 1:ndom) {
          if (type=="mt" & !is.null(barlab)){
            if (!(ncol(barlab[[d]]) %in% c(1,nplot)) )
                msg.stop("When a matrix, bar.labels should have one column per group")
          }
    }

    ngdplot <- nplot * ndom
    ## Compute MDsequences when needed
    if (type == "f") dom.crit <- 0
    if (type == "f" || (!is.null(sortv) && length(sortv) == 1)) {
            if (dom.crit == 0){ ## MD sequences
                		## Calling appropriate function and plotting
        		flist <- names(formals(seqMD))
        		if ("with.missing" %in% names(oolist)) {
        			with.missing <- oolist[["with.missing"]]
        		} else if ("with.missing" %in% flist) {
        			with.missing <- formals(seqMD)$with.missing
        		}

                MDseq <- suppressMessages(seqMD(channels, what="MDseq", with.missing=with.missing, ch.sep="+"))
            }
            else if (dom.crit > 0)
                MDseq <- channels[[dom.crit]]
    }
    else MDseq <- NA

    ## if sortv is a string (except "mds"), we compute sortv on one domain or the MD sequences
	if (!is.null(sortv)) {
        if (length(sortv)==1 && sortv %in% c("from.start", "from.end")) {

            if (dom.crit >= 0) {
                MDext <- unname(as.data.frame(MDseq))
                mxl <- max(seqlength(MDseq))
            }
            else if (dom.crit == -1){
                MDext <- unname(as.data.frame(channels[[1]]))
                for (d in 2:ndom)
                    MDext <- cbind(MDext,unname(as.data.frame(channels[[d]])))
                mxl <- ncol(MDext)

            }
            else if (dom.crit == -2){ ##
                mxl <- max(sapply(channels,function(x){max(seqlength(x))}))

                MDext <- as.data.frame(matrix(nrow=ncase,ncol=ndom*mxl))
                cdom <- seq(1,ndom*mxl,by=ndom)
                for (d in 1:ndom){
                    MDext[,cdom[1:max(seqlength(channels[[d]]))]] <- unname(as.data.frame(channels[[d]]))
                    cdom + 1
                }
            }
            else
                msg.stop("invalid dom.crit value", dom.crit)

            end <- if (sortv=="from.end") { mxl } else { 1 }
    		beg <- if (sortv=="from.end") { 1 } else { mxl }

        	#sortv <- order(do.call(order, unname(as.data.frame(MDseq))[,end:beg]))
        	sortv <- order(do.call(order, MDext[,end:beg]))
        	#x <- x[sortv,]
        } else if (length(sortv)!=ncase) {
            if (type != "rf"){
              msg.stop("sortv must either contain one value for each sequence ",
                "or be one of 'from.start' and 'from.end'")
            } else if (length(sortv) == 1 && sortv != "mds") {
              msg.stop("sortv must either contain one value for each sequence ",
                "or be one of 'mds', 'from.start', and 'from.end'")
            }
        } else {
        	if (is.factor(sortv)) { sortv <- as.integer(sortv) }
        	#x <- x[order(sortv),]
        }

        if (length(sortv)==1 && sortv == "mds") {
          if (type=="rf"){
            if (is.null(diss))
                msg.stop("'diss' required for rf plots")
        	with.missing <- TRUE

##         	if ("sortv" %in% names(oolist))
##                 sortv <- oolist[["sortv"]]
##             else
##                 sortv <- "mds"
##            if (length(sortv)==1 && sortv=="mds"){
                weighted <- TRUE
                if ("weighted" %in% names(oolist)) weighted <- oolist[["weighted"]]
                if (weighted) {
                   if ("weights" %in% names(oolist))
                     weights <- oolist[["weights"]]
                   else
                     weights <- attr(channels[[1]],"weights")
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
            #}
          }
        }
    }


    if (type == "f"){
		seqlist <- suppressMessages(seqformat(MDseq, from = "STS", to = "SPS",
		  SPS.out = list(xfix = "", sdsep = "/"), compress = TRUE))

        weighted <- if ("weighted" %in% names(oolist)) oolist[["weighted"]] else TRUE

    }


	## ===================
	## Defining the layout
	## ===================
	if (type=="Ht" | type =="pc") { with.legend=FALSE }

	## If xaxis argument is provided
	## it interferes with axes argument
## 	if ("xaxis" %in% names(oolist)) {
## 		tmpxaxis <- oolist[["xaxis"]]
## 		if (tmpxaxis==TRUE) {axes<-"all"}
## 		else if (tmpxaxis==FALSE) {axes<-FALSE}
## 		oolist <- oolist[!names(oolist) %in% "xaxis"]
## 	}
##
## 	if ("yaxis" %in% names(oolist)) {
## 		tmpyaxis <- oolist[["yaxis"]]
## 		if (tmpyaxis==TRUE) {yaxes<-"all"}
## 		else if (tmpyaxis==FALSE) {yaxes<-FALSE}
## 		oolist <- oolist[!names(oolist) %in% "yaxis"]
## 	}



	#if (use.layout | !is.null(group) ) {  ##MD for MD plot use.layout can only be TRUE
		## Saving graphical parameters
		savepar <- par(no.readonly = TRUE)


        ##MD use.rf.layout=TRUE not allowed for MD plot, so we comment out
        #if (use.rf.layout) {
		#  lout <- TraMineR.setlayout(2, 1, 2, with.legend, axes, legend.prop)
        #  lout$widths[1] <- 1.3*lout$widths[1]
	  	#  layout(lout$laymat, heights=lout$heights, widths=lout$widths)
        #}
        #else {

        if (dom.byrow) {
            rows <- ndom
            cols <- nplot
        } else {
            rows <- nplot
            cols <- ndom
        }
		lout <- TraMineR.setlayout(ngdplot, rows, cols, with.legend, axes, legend.prop, dom.byrow)
	  	layout(lout$laymat, heights=lout$heights, widths=lout$widths)
        #}
		## Should axis be plotted or not ?
		#xaxis <- 1:ngdplot==lout$axisp

		legpos <- lout$legpos
# 	}
# 	else {
# 		if (axes!=FALSE) {xaxis <- TRUE}
# 		else {xaxis <- FALSE}
# 		legpos <- NULL
# 	}

	## ========
	## Plotting
	## ========
  if ("col.entr" %in% names(oolist)){
      col.entr <- oolist[["col.entr"]]
      oolist <- c(list(col=col.entr), oolist[!"col.entr" %in% names(oolist)])
  }
  for (d in 1:ndom) {
    seqdata <- channels[[d]]

	for (np in 1:nplot) {
        npd <- (d-1)*nplot + np
		## Storing ... arguments in a list
		olist <- oolist

        if (nplot>1)
            maind <- paste0(main,levels(group)[np],": ",dnames[d])
        else
            maind <- paste0(main,dnames[d])
        if (dom.byrow){
            #maind <- if (np==1) main[d] else NA
            #ylabd <- if (np==1) ylab else NA
            ylabd <- ylab[np]
            if (axes=="bottom")
                axesd <- if (d==ndom) TRUE else FALSE
            else if (axes=="all")
                axesd <- TRUE
            else
                axesd <- FALSE

            if (yaxes=="left")
                yaxesd <- if (np==1) TRUE else FALSE
            else if (yaxes=="all")
                yaxesd <- TRUE
            else if (yaxes=="pct")
                yaxesd <- "pct"
            else if (yaxes=="left.pct")
                yaxesd <- if (np==1) "pct" else FALSE
            else
                yaxesd <- FALSE

        }
        else {
            #maind <- if (d==1) main[np] else NA
            ylabd <- if (d==1) ylab[np] else NA
            if (axes=="bottom")
                axesd <- if (np==nplot) TRUE else FALSE
            else if (axes=="all")
                axesd <- TRUE

            else
                axesd <- FALSE

            if (yaxes=="left")
                yaxesd <- if (d==1) TRUE else FALSE
            else if (yaxes=="all")
                yaxesd <- TRUE
            else if (yaxes=="pct")
                yaxesd <- "pct"
            else if (yaxes=="left.pct")
                yaxesd <- if (d==1) "pct" else FALSE
            else
                yaxesd <- FALSE

        }


		plist <- list(main=maind, cpal=cpal.dom[[d]], missing.color=missing.color,
			ylab=ylabd, yaxis=yaxesd, xaxis=axesd,
			xtlab=xtlab, cex.axis=cex.axis)

		## Selecting sub sample for x
		## according to 'group'
		subdata <- seqdata[gindex[[np]],]

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

            ## modifying MD seqtab outcome of the group for current domain
            MDtab <- seqtab(MDseq[gindex[[np]],], weighted = weighted)
            res <- seqdata[gindex[[np]],][attr(MDtab,"idxf"),]
            ## assigning attributes
           	class(res) <- c("stslist.freq",class(res))
        	attr(res, "weights") <- attr(MDtab, "weights")
        	attr(res, "freq") <- attr(MDtab,"freq")
        	attr(res, "nbseq") <- attr(MDtab,"nbseq")
        	attr(res, "weighted") <- attr(MDtab,"weighted")
        	attr(res, "idx") <- attr(MDtab,"idxf")
        	attr(res, "idxf") <- attr(MDtab,"idxf")
        	attr(res, "format") <- attr(MDtab,"format")


			with.missing <- TRUE
			f <- seqtab  ##
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
			with.missing <- TRUE

			## Selecting sub sample for sort variable
			## according to 'group'
			if ("sortv" %in% names(olist)) {
				if (!length(sortv)==1) {
					olist[["sortv"]] <- sortv[gindex[[np]]]
				}
			}

			if (!"space" %in% names(olist)) {olist <- c(olist, list(space=0))}
			if (!"border" %in% names(olist)) {olist <- c(olist, list(border=NA))}
			## Selecting distances according to group
			olist[["diss"]] <- diss[gindex[[np]],gindex[[np]]]
            plist[["skipar"]] <- TRUE

            ##if (use.rf.layout) olist[["which.plot"]] <- "medoids"
		}
		## Mean times
		else if (type=="mt") {
          f <- seqmeant
          plist <- plist[!names(plist) %in% "missing.color"]
          if (!is.null(barlab[[d]])) {
            if (ncol(barlab[[d]])==1)
              olist[["bar.labels"]] <- as.vector(barlab[[d]])
            else
              olist[["bar.labels"]] <- as.vector(barlab[[d]][,np])
          }
        }
		## Modal states
		else if (type=="ms") {
			f <- seqmodst
			plist <- plist[!names(plist) %in% "missing.color"]
		}
		## Representative sequence
		else if (type=="r") {
			f <- seqrep
			with.missing <- TRUE

            if (which.stats == "all")
                stats <- TRUE
            else
                stats <- (which.stats == "first" & d == 1)
            plist[["stats"]] <- stats
			## Removing unused arguments
			plist <- plist[!names(plist) %in% "yaxis"]

			## Selecting distances according to group
			# fixed for deprecated dist.matrix

			olist[["diss"]] <- diss[gindex[[np]],gindex[[np]]]

        # FIXME dist.matrix is deprecated
            olist <- olist[names(olist) != "dist.matrix"]

        }
        else if (type == "pc") { # modification of Reto Bürgin 16.08.2012

                        #plist[["main"]] <- main
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
		if (type != "f") res <- do.call(f, args=fargs)

		olist <- olist[!match.args]
    ## suppress non plot arguments if necessary
        olist <- olist[!names(olist) %in% c("with.missing")]
        if (!(type %in% c("i","I","rf"))) olist <- olist[!(names(olist) %in% c("sortv","weighted"))]
        if (type != "r") olist <- olist[!(names(olist) %in% c("dmax","stats"))]

		plist <- c(list(x=res), plist, olist)
		do.call(plot, args=plist)

        #if (use.rf.layout){
        #    plist[["which.plot"]] <- "diss.to.med"
        #    plist[["ylab"]] <- NA
        #    plist[["yaxis"]] <- FALSE
        #    do.call(plot, args=plist)
        #}
	}

	## Plotting the legend
	if (!is.null(legpos)) {
		## Extracting some sequence characteristics
		nr <- attr(seqdata,"nr")

		if (is.null(ltext.dom[[d]]))
            ltext <- attr(seqdata,"labels")
        else
            ltext <- ltext.dom[[d]]

		if (is.null(missing.color)) missing.color <- attr(seqdata,"missing.color")

		if (is.null(cpal.dom[[d]]))
            cpal <- attr(seqdata,"cpal")
        else
            cpal <- cpal.dom[[d]]

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

  }
	## Restoring graphical parameters
	#if (use.layout | !is.null(group))
    {par(savepar)}
}
