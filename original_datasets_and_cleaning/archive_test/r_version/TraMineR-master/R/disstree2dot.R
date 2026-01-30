###########################
## Internal plot method
###########################
DTNplotfunc <- function(image.data, plot.fun, plot.fun.cex.main,
				plot.fun.use.title=TRUE, plot.fun.label.pos="main",
				plot.fun.node.pos="main", plot.fun.split.pos="sub",
				plot.fun.title.outer=FALSE, ...) {
	getSize <- function(loc){
		locvect <- c(plot.fun.label.pos,  plot.fun.node.pos, plot.fun.split.pos)
		return(sum(as.integer(locvect==loc))*plot.fun.cex.main)

	}
	if (plot.fun.use.title) {
		top <- getSize("main")
		bottom <- getSize("sub")
		left <- getSize("ylab")
		if(!plot.fun.title.outer){
			par(mar=c(bottom, left, top, 0), font.sub=2, mgp=c(0, 0, 0))
		}else{
			par(oma=c(bottom, left, top, 0), font.sub=2, mgp=c(0, 0, 0))
		}
		#on.exit(par(oldpar))
	}
	plot.fun(image.data, ...)
}

DTNseqplot <- function(ind, seqdata, sortv=NULL, diss=NULL, ...) {
	if (!is.null(sortv)){
		if(length(sortv) > 1) {
            seqplot(seqdata[ind, ], sortv=sortv[ind], ...)
        } else {
            seqplot(seqdata[ind, ], sortv=sortv, ...)
        }
	} else if(!is.null(diss)){
		seqplot(seqdata[ind, ], diss=diss[ind,ind], ...)
	}
	else {
		seqplot(seqdata[ind, ], ...)
	}
}


DTNseqlegend <- function(filename, seqdata, cex.legend, with.legend, image.format="jpg", ...) {
	image.legend <- NULL
	arguments <- list(...)
	if(with.legend!=FALSE) {
		if(image.format!="jpg"){
			device <- "png"
			image.fmt <- "png"
		}
		else {
			device <- "jpeg"
			image.fmt <- "jpg"
		}

		if (is.null(arguments[["device.args"]])) {
			device.args <- list()
		}
		else {
			device.args <- arguments[["device.args"]]
		}
		image.legend <- paste(filename,"legend", image.fmt, sep=".")
		device.args$file <- image.legend

		do.call(device, device.args)
		plot.new()
		seqlegend(seqdata, cex=cex.legend, title="Legend", position="center",  bty="n")
		dev.off()
	}
	return(image.legend)
}

disstreedisplay <- function(tree, filename = NULL, image.data = NULL,
  image.fun = plot, only.leaf = FALSE, cex.main = 3, image.format = "png",
  with.quality = TRUE, cex.quality = cex.main, legend.text = NULL,
  show.tree = TRUE, show.depth = FALSE, imagedata, imagefunc, imgLeafOnly,
  title.cex, imageformat, withquality, quality.fontsize, legendtext, showtree,
  showdepth, ...) {

  TraMineR.check.depr.args(alist(image.data = imagedata, image.fun = imagefunc,
    only.leaf = imgLeafOnly, cex.main = title.cex, image.format = imageformat,
    with.quality = withquality, cex.quality = quality.fontsize,
    legend.text = legendtext, show.tree = showtree, show.depth = showdepth))

	actualdir <- getwd()
	tmpdir <- tempdir()
	tmpdisstree <- basename(tempfile(pattern="tmpdisstree"))
	on.exit(setwd(actualdir))
	setwd(tmpdir)
	disstreedisplayInternal(tree=tree, filename=filename, tmpdisstree=tmpdisstree, image.data=image.data, image.fun=image.fun,
							only.leaf=only.leaf,cex.main=cex.main, image.format=image.format, with.quality=with.quality,
							cex.quality=cex.quality, legend.text=legend.text, show.tree=show.tree, show.depth=show.depth, ...)
	setwd(actualdir)
	if(!is.null(filename)){
		success <- file.copy(file.path(tmpdir, paste(tmpdisstree, image.format, sep=".")), filename, overwrite=TRUE)
		if ( !success )
			stop("Cannot copy tmpdisstree.",image.format," as ",filename,"!")
	}

}

disstreedisplayInternal <- function(tree, filename, tmpdisstree, image.data, image.fun, only.leaf, cex.main, image.format,
							with.quality, cex.quality, legend.text, show.tree, show.depth, gvpath=NULL, ...) {
	if(image.format!="jpg"){
		disstree2dotp(tree=tree, filename=tmpdisstree, image.data=image.data, only.leaf=only.leaf, image.fun=image.fun,
			cex.main=cex.main, device="png", image.format="png", legend.text=legend.text, with.quality=with.quality, cex.quality=cex.quality, show.depth=show.depth, ...)
	}
	else {
		disstree2dotp(tree=tree, filename=tmpdisstree, image.data=image.data, only.leaf=only.leaf, image.fun=image.fun,
			cex.main=cex.main, legend.text=legend.text, with.quality=with.quality, cex.quality=cex.quality, show.depth=show.depth, ...)
	}

	myshellrun <- function(cmd, gvpath=NULL, ...) {
		cmd <- paste(gvpath, cmd, sep="")
		if (.Platform$OS.type=="windows") {
			return(system(paste(Sys.getenv("COMSPEC"),"/c \"", cmd, "\""), ...))
		}
		else {
			return(system(cmd,...))
		}
	}

	if(image.format!="jpg"){
		dotcommand <- paste(" -Tpng -o", tmpdisstree,".png ", tmpdisstree,".dot", sep="")
	}else{
		dotcommand <- paste(" -Tjpg -o", tmpdisstree,".jpg ", tmpdisstree,".dot", sep="")
	}

	if (is.null(gvpath)) {
		## First check for GVPATH
		gvpath <- Sys.getenv("GVPATH")
		if(gvpath==""){
			getGraphVizPath <- function(envVar){
				envDir <- Sys.getenv(envVar)
				dd <- dir(envDir, "Graphviz(.*)")
				if(length(dd)>0){
					cond <- file.exists(paste(envDir, dd, "bin", "dot.exe", sep="\\"))
					if(sum(cond)==0){
						return(NULL)
					}
					dd <- dd[cond]
					versions <- sub("Graphviz[[:space:]]?", "", dd)
					if (versions != ""){
						ii <- which.max(as.numeric(versions))
						message(" [>] GraphViz version ", versions[ii], " found.")
						dd <- dd[ii]
					}
					else {
						message(" [>] GraphViz path found.")
					}
					return(paste(envDir, dd, sep="\\"))
				}
				return(NULL)
			}
			## GVPATH not found, check default locations directories...
			gvpath <- getGraphVizPath("programfiles")
			if(is.null(gvpath)){
				gvpath <- getGraphVizPath("programfiles(x86)")
			}

		}
	}
	#gvpath <- paste("\"", gvpath, "Graphviz\\bin\\dot.exe\" ", sep="")
	gvpath <- paste("\"", gvpath, "\\bin\\dot.exe\" ", sep="")
	if (.Platform$OS.type!="windows") {
		gvpath <- "dot "
	}
	dotval <- myshellrun(dotcommand, gvpath=gvpath)
	if(dotval==1){
		stop(paste(" [!] GraphViz was not found. If you haven't, please install GraphViz to use this function: see http://www.graphviz.org",
			 " [!] If GraphViz is installed on your computer, you need to specify the GraphViz installation directory using the argument gvpath='installdir'",
			 " [!] You can also add this directory to the PATH environment variable",
			 " [!] GraphViz installation directory usually looks like 'C:\\Program Files\\GraphViz'\n", sep="\n"
			 ))
	}


	if (!(image.format %in% c("jpg", "png"))) {
		imagickval <- myshellrun(paste("convert ", tmpdisstree,".png ",tmpdisstree,".", image.format, sep=""))
		if (imagickval == 1) {
			stop("To use another format than jpeg or png, you should install ImageMagick: see http://www.imagemagick.org")
		}
	}
    if (show.tree) {
    	if (.Platform$OS.type=="windows") {
    		myshellrun(paste("start ", tmpdisstree, ".", image.format, sep=""), wait=FALSE)
    	}
    	else if(Sys.info()[1]=="Darwin"){
			myshellrun(paste("open ", tmpdisstree, ".", image.format, sep=""), wait=FALSE)
		}
    	else {
    		myshellrun(paste("display ", tmpdisstree, ".", image.format, sep=""), wait=FALSE)
    	}
    }
	
	return(invisible())

}

seqtreedisplay <- function(tree, filename = NULL, seqdata = tree$info$object,
  only.leaf = FALSE, sortv = NULL, diss = NULL, cex.main = 3, with.legend = "auto",
  #cex.legend = cex.main, axes = FALSE, image.format = "png", with.quality = TRUE,
  cex.legend = cex.main, xaxis = FALSE, image.format = "png", with.quality = TRUE,
  cex.quality = cex.main, legend.text = NULL, show.tree = TRUE, show.depth = FALSE,
  imgLeafOnly, dist.matrix, title.cex, withlegend, legend.fontsize, imageformat,
  withquality, quality.fontsize, legendtext, showtree, showdepth, axes, ...) {

  TraMineR.check.depr.args(alist(only.leaf = imgLeafOnly, diss = dist.matrix,
    cex.main = title.cex, with.legend = withlegend, cex.legend = legend.fontsize,
    image.format = imageformat, with.quality = withquality,
    cex.quality = quality.fontsize, legend.text = legendtext, show.tree = showtree,
    show.depth = showdepth, xaxis=axes))

	actualdir <- getwd()
	tmpdir <- tempdir()
	tmpdisstree <- basename(tempfile(pattern="tmpseqtree"))
	on.exit(setwd(actualdir))
	setwd(tmpdir)

	image.legend <- DTNseqlegend(filename=tmpdisstree, seqdata=seqdata, cex.legend=cex.legend, with.legend=with.legend, image.format=image.format, ...)
	if(!is.null(diss)){
		diss <- as.matrix(diss)
	}
	disstreedisplayInternal(tree=tree, filename=filename, tmpdisstree=tmpdisstree, image.data=NULL, image.fun=DTNseqplot,
							only.leaf=only.leaf, cex.main=cex.main, image.format=image.format, with.quality=with.quality,
							cex.quality=cex.quality, legend.text=legend.text, show.tree=show.tree, show.depth=show.depth, image.legend=image.legend,
							seqdata=seqdata, sortv=sortv, diss=diss, xaxis=xaxis, with.legend=FALSE, ...)
	setwd(actualdir)
	if(!is.null(filename)){
		success <- file.copy(file.path(tmpdir, paste(tmpdisstree, image.format, sep=".")), filename, overwrite=TRUE)
		if ( !success )
			stop("Cannot copy tmpdisstree.",image.format," as ",filename,"!")
	}
	return(invisible())
}

###########################
## Shortcut to build sequences tree
###########################
seqtree2dot <- function(tree, filename, seqdata = tree$info$object,
  only.leaf = FALSE, sortv = NULL, diss = NULL, cex.main = 3, with.legend = "auto",
  #cex.legend = cex.main, with.quality = FALSE, cex.quality = cex.main, axes = FALSE,
  cex.legend = cex.main, with.quality = FALSE, cex.quality = cex.main, xaxis = FALSE,
  #imgLeafOnly, dist.matrix, title.cex, withlegend, withquality, ...) {
  imgLeafOnly, dist.matrix, title.cex, withlegend, withquality, axes, ...) {

  TraMineR.check.depr.args(alist(only.leaf = imgLeafOnly, diss = dist.matrix,
    cex.main = title.cex, with.legend = withlegend, with.quality = withquality
	, xaxis=axes))

	image.legend <- DTNseqlegend(filename=filename, seqdata=seqdata, cex.legend=cex.legend, with.legend=with.legend, ...)
	if(!is.null(diss)){
		diss <- as.matrix(diss)
	}
	disstree2dotp(tree, filename, image.data=NULL, only.leaf=only.leaf, seqdata=seqdata, cex.main=cex.main,
			#sortv=sortv,diss=diss, image.fun=DTNseqplot, with.legend=FALSE, axes=axes,
			sortv=sortv,diss=diss, image.fun=DTNseqplot, with.legend=FALSE, xaxis=xaxis,
			image.legend=image.legend, with.quality=with.quality, cex.quality=cex.quality, ...)
}


###########################
## Generate dot file
###########################
disstree2dotp <- function(tree, filename, image.data = NULL, only.leaf = FALSE,
  image.fun = plot, cex.main = 3, with.quality = TRUE, cex.quality = cex.main,
  title.outer = FALSE, imagedata, imgLeafOnly, imagefunc, title.cex, withquality,
  quality.fontsize, ...) {

  TraMineR.check.depr.args(alist(image.data = imagedata, only.leaf = imgLeafOnly,
    image.fun = imagefunc, cex.main = title.cex, with.quality = withquality,
    cex.quality = quality.fontsize))

	image.quality <- NULL
	arguments <- list(...)
	if(with.quality) {
		device <- ifelse(is.null(arguments[["device"]]), "jpeg", arguments[["device"]])
		image.format <- ifelse(is.null(arguments[["image.format"]]), "jpg", arguments[["image.format"]])
		if (is.null(arguments[["device.args"]])) {
			device.args <- list()
		}
		else {
			device.args <- arguments[["device.args"]]
		}
		image.quality <- paste(filename,"quality", image.format, sep=".")
		device.args$file <- image.quality
		do.call(device, device.args)
		par(mar=rep(0.2, 4))
		plot.new()
		plot(0, type = "n", axes = FALSE, xlab = "", ylab = "")
		rowStat <- function(x, n){
			formd <- function (x){
				return(format(x, digits =getOption("digits")-2))
			}
			star <- function(val){
				return(as.character(cut(val, c(0, 0.001, 0.01, 0.05, 1), labels=c("***", "**", "*",""))))
			}
			return(paste(n,": ", formd(x$info$adjustment$stat[n,1])," ", star(x$info$adjustment$stat[n,2]), sep=""))
		}
		ltext <- c(rowStat(tree, "Pseudo F"), rowStat(tree, "Pseudo R2"), rowStat(tree, "Levene"))
		legend("center", legend = ltext, cex = cex.quality, title="Global quality", bty="n")
		dev.off()
	}
	disstree2dot(tree, filename, image.data=image.data, only.leaf=only.leaf, cex.main=cex.main, image.fun=DTNplotfunc, plot.fun=image.fun,
			use.title=TRUE, label.pos="main", node.pos="main", split.pos="sub",
			plot.fun.use.title=TRUE, plot.fun.label.pos="main", plot.fun.node.pos="main",
			plot.fun.split.pos="sub", plot.fun.cex.main=3, image.quality=image.quality,
			title.outer=title.outer, plot.fun.title.outer=title.outer, ...)
			

}

###########################
## Generate dot file
###########################
disstree2dot <- function(tree, filename, digits = 3, image.fun = NULL,
  image.data = NULL, only.leaf = FALSE, device = "jpeg", image.format = "jpg",
  device.args = list(), use.title = TRUE, label.pos = "main", node.pos = "main",
  split.pos = "sub", cex.main = 1, legend.text = NULL, image.legend = NULL,
  image.quality = NULL, show.depth = FALSE, title.outer = FALSE, imagefunc,
  imagedata, imgLeafOnly, devicefunc, imageext, device.arg, label.loc, node.loc,
  split.loc, title.cex, legendtext, legendimage, qualityimage, showdepth, ...) {

  TraMineR.check.depr.args(alist(image.fun = imagefunc, image.data = imagedata,
    only.leaf = imgLeafOnly, device = devicefunc, image.format = imageext,
    device.args = device.arg, label.pos = label.loc, node.pos = node.loc,
    split.pos = split.loc, cex.main = title.cex, legend.text = legendtext,
    image.legend = legendimage, image.quality = qualityimage, show.depth = showdepth))

	dotfile <- paste(filename, ".dot", sep="")
	node <- tree$root
	cat("digraph distree{\n", file=dotfile)
	if (!is.null(image.legend)) {
		str <- paste("\"node_image.legend\"[shape=box, image=\"", image.legend,"\", imagescale=true, label=\" \"", sep="")
		cat(paste(str, "];\n", sep=""), file=dotfile, append=TRUE)
	}


	if (!is.null(image.fun) && is.null(image.data)) {
		image.data <- as.data.frame(node$info$ind)
	}
	formd <- function (x){
		return(format(x, digits =digits))
	}
	nodedepthranking <- list()
	DTN2DotInternal <- function(preced, node, pos, label) {
		nodename <- paste(preced, "_", pos, sep="")
		nodedepthranking[[nodename]] <<- node$info$splitschedule
		stringcontentnode <- paste("n: ", formd(node$info$n), " s2: ",
				formd(node$info$vardis), sep="")
		if (!is.null(node$split)) {
			stringcontentsplit <- paste("Split: ", colnames(tree$data)[node$split$varindex], " R2:",
					formd(node$split$info$R2), sep="")
		} else {
			stringcontentsplit <- ""
		}
		if (!is.null(image.fun) && (!only.leaf || is.null(node$split))) {
			device.args$file <- paste(nodename, image.format, sep=".")
			do.call(device, device.args)
			if (!is.null(ncol(image.data))) {
				image.fun(image.data[node$info$ind, ], ...)
			} else {
			  	image.fun(image.data[node$info$ind], ...)
			}
			if (use.title) {
				title.arg <- list(main = NULL, sub = NULL, xlab = NULL, ylab = NULL,
							line = NA, outer = title.outer, cex.main=cex.main,	cex.sub=cex.main,
							cex.lab=cex.main)
				title.arg[[node.pos]] <- stringcontentnode
				title.arg[[split.pos]] <- stringcontentsplit
				title.arg[[label.pos]] <- label
				if (label.pos==node.pos) {
					title.arg[[label.pos]] <- paste(label, stringcontentnode, sep="\n")
				}
				if (label.pos==split.pos) {
					title.arg[[label.pos]] <- paste(title.arg[[label.pos]], stringcontentsplit, sep="\n")
				}
				if (node.pos==split.pos) {
					if (label.pos!=split.pos) {
						title.arg[[node.pos]] <- paste(stringcontentnode, stringcontentsplit, sep="\n")
					}
				}
				do.call("title", title.arg)
			}
			dev.off()
			imgstr <- paste(" image=\"", nodename,".", image.format,"\", imagescale=true, ", sep="")
		}
		else {
			imgstr <- ""
		}
		if (!use.title) {
			str <- paste("\"", nodename, "\"[shape=box, ", imgstr, " label=", "\"",
					stringcontentnode, stringcontentsplit, "\"", sep="")
		}
		else {
			str <- paste("\"", nodename, "\"[shape=box, ", imgstr, " label=\" \"", sep="")
		}
		cat(paste(str, "];\n", sep=""), file=dotfile, append=TRUE)
		split_print <- function(sp){
			str_split <- character(2)
			if (!is.null(sp$breaks)) {
				str_split[1] <- paste("<=", formd(sp$breaks))
				str_split[2] <- paste(">", formd(sp$breaks))
			}
			else {
				## lgrp <- levels(factor(tree$data[,sp$varindex]))
				str_split[1] <- paste("[", paste(sp$labels[sp$index==1], collapse=","),"]")
				str_split[2] <- paste("[", paste(sp$labels[sp$index==2], collapse=","),"]")
			}
			if(!is.null(sp$naGroup)){
				str_split[sp$naGroup] <- paste(str_split[sp$naGroup], "with NA")
			}
			return(str_split)
		}
		if (!is.null(node$split)) {
			str_split <- split_print(node$split)
			pos <- c("left", "right")
			for (i in 1:2) {
				DTN2DotInternal(preced=nodename, node=node$kids[[i]], pos=pos[i], label=str_split[i])
			}
		}
	}

	DTN2DotInternalRelation <- function(preced, node, pos){
		nodename <- paste(preced, "_", pos, sep="")
		if (node$info$depth!=1) {
			cat(paste("\"", preced, "\"->", "\"", nodename, "\";\n", sep=""), file=dotfile, append=TRUE)
		}
		if (!is.null(node$split)) {
			pos <- c("left", "right")
			for (i in 1:2) {
				DTN2DotInternalRelation(preced=nodename, node=node$kids[[i]], pos=pos[i])
			}
		}


	}

	DTN2DotInternal(preced=filename, node=node, pos="none", label="Root")
	if (!is.null(image.quality)) {
		str <- paste("\"node_qualityimage\"[shape=box, image=\"", image.quality,"\", imagescale=true, label=\" \"", sep="")
		cat(paste(str, "];\n", sep=""), file=dotfile, append=TRUE)
	}
	if (!is.null(legend.text)) {
		str <- paste("\"node_legend.text\"[shape=box, label=<", legend.text, ">", sep="")
		cat(paste(str, "];\n", sep=""), file=dotfile, append=TRUE)
	}
	if(show.depth){
		fontsize <- paste("[shape=box, fontsize=", cex.main*20, "];\n", sep="")
		unikrank <- unlist(unique(nodedepthranking))
		cat(paste(unikrank, collapse=fontsize),fontsize, file=dotfile, append=TRUE)
		nodename <- names(nodedepthranking)
		for(rank in unikrank){
			cat("{ rank=same ;", rank, ";", paste(nodename[nodedepthranking==rank], collapse="; "), ";}\n", file=dotfile, append=TRUE)
		}
		cat(paste(sort(unikrank), collapse=" -> "), ";\n", file=dotfile, append=TRUE)
	}
	DTN2DotInternalRelation(preced=filename, node=node, pos="none")


	cat("}\n", file=dotfile, append=TRUE)

}
