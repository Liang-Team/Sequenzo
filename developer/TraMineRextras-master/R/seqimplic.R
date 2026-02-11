###########################
## Compute the implicative statistic of a rule
###########################


implicativestat <- function(x, y, type="intensity", resid="standard", weights=NULL, continuity=FALSE) {
	if (!(type %in% c("intensity", "indice"))) {
		stop("type should be intensity or indice")
	}
	if (!(resid %in% c("standard", "deviance", "Freeman-Tukey", "adjusted"))) {
		stop("resid should be one of standard, deviance, Freeman-Tukey or ajusted")
	}
	if(is.null(weights)){
		weights <- rep(1, length(x))
	}
	x <- factor(x)
	y <- factor(y)
	xgrp <- levels(x)
	ygrp <- levels(y)
	result <- matrix(0, nrow=length(xgrp), ncol=length(ygrp))
	n <- sum(weights)
	rownames(result) <- xgrp
	colnames(result) <- ygrp
	for (i in 1:length(xgrp)) {
		condi <- x==xgrp[i]
		for (j in 1:length(ygrp)) {
			condj <- y==ygrp[j]
			Nnbj <- sum(weights[(condi)&!condj])
			if(continuity) {
				Nnbj <- Nnbj + 0.5
			}
			Nexpnbj <- sum(weights[condi])*sum(weights[!condj])/n
			if (resid=="standard") {
				indice <- (Nnbj-Nexpnbj)/sqrt(Nexpnbj)
			}
			else if (resid=="deviance") {
				indice <- sign(Nnbj-Nexpnbj)*sqrt(abs(2*Nnbj*log(Nnbj/Nexpnbj)))
			}
			else if (resid=="Freeman-Tukey") {
				indice <- sqrt(Nnbj)+sqrt(1+Nnbj)-sqrt(4*Nexpnbj+1)
			}
			else if (resid=="adjusted") {
				indice <- (Nnbj-Nexpnbj)/sqrt(Nexpnbj*(sum(weights[condj])/n)*(1-sum(weights[condi])/n))
			}
						
			
			if (type=="indice") {
				result[i, j] <- indice
			}
			else {
				result[i, j] <- pnorm(-indice)
			}
		}
	}
	return(result)
}
	
seqimplic <- function(seqdata, group, with.missing=FALSE, weighted=TRUE, na.rm=TRUE) {
	type <- "indice"
	resid <- "adjusted"
	continuity <- TRUE
	if (!inherits(seqdata, "stslist")){
        stop(" [!] seqdata is not a sequence object, use seqdef function to create one")
	}
	ret <- list(type=type, resid=resid, with.missing=with.missing, continuity=continuity)

	seqg <- FALSE
	if(na.rm){
		cond <- !is.na(group)
		group <- factor(group[cond])
		seqdata <- seqdata[cond, ]
	} else {
		group <- factor(group)
	}
	ret$levels <- levels(group)

	alph <- alphabet(seqdata)
	cpal <- cpal(seqdata)
	ret$labels <- attr(seqdata, "labels")
	discardMissing <- FALSE
	if(with.missing){
		alph <- c(alph, attr(seqdata, "nr"))
		cpal <- c(cpal, attr(seqdata, "missing.color"))
		ret$labels <- c(ret$labels, "Missing")
	}
	else if(any(seqdata==attr(seqdata, "nr"))){
		discardMissing <- TRUE
		warning("Discarding missing values")
	}
	ret$alphabet <- alph
	ret$cpal <- cpal
	weights <- attr(seqdata, "weights")
	if(is.null(weights) || !weighted){
		weights <- rep(1, nrow(seqdata))
	}
	ret$indices <- array(NA, dim=c(length(ret$levels), length(alph), ncol(seqdata)), dimnames=list(ret$levels, alph, colnames(seqdata)))
	
	for(i in 1:ncol(seqdata)){
		seqi <- seqdata[, i]
		if(discardMissing) {
			cond <- !seqi %in% c(attr(seqdata, "nr"), attr(seqdata, "void"))
		} else {
			cond <- seqi != attr(seqdata, "void")
		}
		# if(seqg){
			# group <- groupseq[, i]
			# if(!na.rm) {
				# group <- recodef(group, list(Missing=c(c(attr(groupseq, "nr"), attr(groupseq, "void")))))
			# }else{
				# cond <- cond &(!group %in% c(attr(groupseq, "nr"), attr(groupseq, "void")))
			# }
		# }
		i_indice  <- implicativestat(x=group[cond], y=seqi[cond], type=type, resid=resid, weights=weights[cond], continuity=continuity)
		ret$indices[ rownames(i_indice) , colnames(i_indice), i] <- i_indice
	}
	class(ret) <- "seqimplic"
	ret$xtstep <- attr(seqdata, "xtstep")
	ret$tick.last <- attr(seqdata, "tick.last")
	return(ret)
}




print.seqimplic <- function(x, xtstep=NULL, tick.last=NULL, round=NULL, conf.level=NULL, na.print="", ...){
	if (is.null(xtstep)) {
		xtstep <- ifelse(!is.null(x$xtstep), x$xtstep, 1)
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(x$tick.last), x$tick.last, FALSE)
	}
	indices <- x$indices
	if(!is.null(round)){
		indices <- round(indices, round)
	}
	if(!is.null(conf.level)){
		if(x$type == "indice"){
			conf.level <- -qnorm(conf.level)
			cond <- indices > conf.level
		} else {
			cond <- indices < conf.level
		}
		indices[cond] <- ""
		indices[!cond] <- "*"
	}
	for(ll in x$levels){
		cat(ll, "\n")
    npos <- length(dimnames(indices)[[3]])
		tpos <- seq(1, npos, xtstep)
    if (tick.last & tpos[length(tpos)] < npos) tpos <- c(tpos,npos)
		print(indices[ll, , tpos], na.print=na.print, ...)
	}
}


plot.seqimplic <- function(x, main=NULL, ylim=NULL, xaxis=TRUE,
	ylab="Implication", yaxis=TRUE, axes="all", xtlab=NULL,
    xtstep = NULL, tick.last = NULL, cex.axis=1,
	with.legend="auto", ltext=NULL, cex.legend=1,
	legend.prop=NA, rows=NA, cols=NA, conf.level=0.95, lwd=1, only.levels=NULL, ...){
	
	savepar <- par(no.readonly = TRUE)
	on.exit(par(savepar))
	if(is.null(only.levels)){
		only.levels <- x$levels
	}
	if (is.null(xtstep)) {
		xtstep <- ifelse(!is.null(x$xtstep), x$xtstep, 1)
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(x$tick.last), x$tick.last, FALSE)
	}

	plotindex <- (1:length(x$levels))[x$levels %in% only.levels]
	nplot <- length(plotindex)
	if(nplot>1||with.legend!=FALSE){
		lout <- TraMineR:::TraMineR.setlayout(nplot, rows, cols, with.legend, axes, legend.prop)
		layout(lout$laymat, heights=lout$heights, widths=lout$widths)
		legpos <- lout$legpos
	}
	x$indices <- -x$indices
	if(is.null(ylim)) {
		if(x$type=="intensity"){
			ylim <- c(0, 1)
		} else {
			ylim <- c(0, max(x$indices, na.rm=TRUE))
			if(any(is.na(ylim)|!is.finite(ylim))){
				ylim <- c(0, 1)
			}
	
		}	
	}
	if(is.null(main)){
		main <- x$levels
	}else if(length(main) == 1) {
		main <- paste(main, x$levels, sep=" - ")
	}
	for (np in plotindex) {
		toplot <-  x$indices[np, , ]
		toplot[toplot<0] <- NA
		plot(toplot[1, ], ylim=ylim, xlim=c(1, dim(x$indices)[3]), col=x$cpal[1], main=main[np],
		type="l", ylab=ylab, axes=FALSE, xlab=NA, lwd=lwd, ...)
		for(a in 2:length(x$alphabet)) {
			lines(toplot[a, ], col=x$cpal[a], type="l", lwd=lwd,  ...)
		}
		## Plotting the x axis
		if (xaxis) {
			if (is.null(xtlab)){
				xtlab <- dimnames(x$indices)[[3]]
			}
      npos <- length(xtlab)
		  tpos <- seq(from=1, to=npos, by=xtstep)
      if (tick.last & tpos[length(tpos)] < npos) tpos <- c(tpos,npos)
			axis(1, at=tpos-0.5, labels=xtlab[tpos], cex.axis=cex.axis, ...)
		}
		if (is.null(yaxis) || yaxis) {
			axis(2, cex.axis=cex.axis, ...)
		}
		if(!is.null(conf.level)) {
			for(conf in conf.level){
				label <- paste("Conf. ", format(conf))
				if(x$type=="intensity"){
					h <- conf
				} else {
					h <- qnorm(conf)				
				}
				abline(h=h, lty=3, col="grey", lwd=lwd)
				cex.textlab <- 0.8*cex.axis
				text(x=dim(x$indices)[3] - strwidth(label, cex=cex.textlab)/2, y=h + strheight(label, cex=cex.textlab), labels=label, cex=cex.textlab)
			}
		}
	}
	
	if (with.legend!=FALSE) {
		## Extracting some sequence characteristics
		TraMineR:::TraMineR.legend(legpos, x$labels, x$cpal, cex=cex.legend)
	}

}
