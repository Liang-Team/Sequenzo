## plot superposed transversal entropies
## author: Gilbert Ritschard

seqplot.tentrop <- function(seqdata, group,
     main="auto", col=NULL, lty=NULL, lwd=3.5, ylim=NULL,
     xtlab=NULL, xtstep=NULL, tick.last=NULL,
     with.legend=TRUE, glabels=NULL, legend.pos="topright",
     horiz=FALSE, cex.legend=1, ...) {

  group <- factor(group)

  if (!is.null(main) && main[1] == "auto") {
        main <- "Tranversal Entropies"
  }


  entrop <- by(seqdata, group, seqstatd)

  k <- length(entrop)
  default.col <- qualitative_hcl(k, palette = "Dark 3")
  ##default.col <- brewer.pal(9,"Set1")
  ##default.col <- c("red","blue","black","magenta","green")
  if(is.null(col)) {
     #col <- colors.list[1:k]
     col <- default.col
  }
  kk <- ceiling(k/length(col))
  col <- rep(col,kk)
  col <- col[1:k]

  default.lty <- c("solid","dashed","dotted")
  if(is.null(lty)) {
     lty <- default.lty
  }
  kk <- ceiling(k/length(lty))
  lty <- rep(lty,kk)
  lty <- lty[1:k]

  kk <- ceiling(k/length(lwd))
  lwd <- rep(lwd,kk)
  lwd <- lwd[1:k]

  npos <- ncol(seqdata)

  if(is.null(xtlab)){
      xtlab <- names(seqdata)
      ##xtlab <- paste("t",1:npos,sep="")
  }

  if(is.null(glabels)){
      glabels <- levels(group)
  }

	if (is.null(xtstep)) {
		xtstep <- ifelse(!is.null(attr(seqdata, "xtstep")), attr(seqdata, "xtstep"), 1)
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(seqdata, "tick.last")), attr(seqdata, "tick.last"), FALSE)
	}


  if(is.null(ylim)){
        maxe <- max(entrop[[1]]$Entropy)
        mine <- min(entrop[[1]]$Entropy)
        for (i in 2:k){
            maxe <- max(maxe,entrop[[i]]$Entropy)
            mine <- min(mine,entrop[[i]]$Entropy)
        }
        ylim <- c(floor(10*mine),ceiling(10*maxe))/10
  }

  plot(0, type= "n", axes=FALSE, xlab="", ylab="Entropy", main=main, ylim=ylim, xlim=c(1,npos), ...)
  for (i in 1:k) {
     lines(entrop[[i]]$Entropy, col=col[i],  type="l", lty=lty[i], lwd=lwd[i], ...)
  }
	tpos <- seq(from=1, to=npos, by=xtstep)
  if (tick.last & tpos[length(tpos)] < npos) tpos <- c(tpos,npos)
  #axis(1,labels=xtlab[tpos],at=tpos, las=las, cex.axis=cex.axis)
  axis(1,labels=xtlab[tpos],at=tpos, ...)
  axis(2, ...)
  if(with.legend){
    legend(legend.pos, legend=glabels,  lwd=lwd, lty=lty[1:k], col=col[1:k], horiz=horiz, cex=cex.legend)
  }

  return(invisible(k))
}


##k <- seqplot.tentrop(seqs.coh, group=seqs$highedu, xtlab=xtlab20)

##

seqplot.tentrop.m <- function(seqdata.list,
     main="auto", col=NULL, lty=NULL, lwd=3.5, ylim=NULL,
     xtlab=NULL, xtstep=NULL, tick.last=NULL,
     with.legend=TRUE, glabels=names(seqdata.list), legend.pos="topright",
     horiz=FALSE, cex.legend=1, ...) {

  ncurve <- length(seqdata.list)
  if (inherits(seqdata.list,"stslist") || !inherits(seqdata.list,"list") || ncurve < 2) {
    stop("seqplot.tentrop.m: seqdata.list must be a list of at least 2 seqlist objects", .call=FALSE)
  }
  warn <- FALSE
  for (i in 2:ncurve){
    if(ncol(seqdata.list[[1]]) != ncol(seqdata.list[[i]])) {warn <- TRUE}
  }
  if (warn) {
    warning("sequence objects in seqdata.list are not all of same length")
    }
  if (!is.null(main) && main[1] == "auto") {
        main <- "Tranversal Entropies"
  }

  entrop <- lapply(seqdata.list, seqstatd)

  k <- length(entrop)
  default.col <- brewer.pal(9,"Set1")
  ##default.col <- c("red","blue","black","magenta","green")
  if(is.null(col)) {
     #col <- colors.list[1:k]
     col <- default.col
  }
  kk <- ceiling(k/length(col))
  col <- rep(col,kk)
  col <- col[1:k]

  default.lty <- c("solid","dashed","dotted","solid","dashed")
  if(is.null(lty)) {
     lty <- default.lty
  }
  kk <- ceiling(k/length(lty))
  lty <- rep(lty,kk)
  lty <- lty[1:k]

  kk <- ceiling(k/length(lwd))
  lwd <- rep(lwd,kk)
  lwd <- lwd[1:k]

  npos <- max(unlist(lapply(seqdata.list,ncol)))

  if(is.null(xtlab)){
      xtlab <- paste("t",1:npos,sep="")
  }

  if(is.null(glabels)){
      glabels <- paste("seq",1:length(seqdata.list),sep="")
  }

	if (is.null(xtstep)) {
		xtstep <- ifelse(!is.null(attr(seqdata.list[[1]], "xtstep")), attr(seqdata.list[[1]], "xtstep"), 1)
	}
	if(is.null(tick.last)){
		tick.last <- ifelse(!is.null(attr(seqdata.list[[1]], "tick.last")), attr(seqdata.list[[1]], "tick.last"), FALSE)
	}

  if(is.null(ylim)){
        maxe <- max(entrop[[1]]$Entropy)
        mine <- min(entrop[[1]]$Entropy)
        for (i in 2:k){
            maxe <- max(maxe,entrop[[i]]$Entropy)
            mine <- min(mine,entrop[[i]]$Entropy)
        }
        ylim <- c(floor(10*mine),ceiling(10*maxe))/10
  }

  plot(0, type= "n", axes=FALSE, xlab="", ylab="Entropy", main=main, ylim=ylim, xlim=c(1,npos), ...)
  for (i in 1:k) {
     lines(entrop[[i]]$Entropy, col=col[i],  type="l", lty=lty[i], lwd=lwd[i], ...)
  }
	tpos <- seq(from=1, to=npos, by=xtstep)
  if (tick.last & tpos[length(tpos)] < npos) tpos <- c(tpos,npos)
  axis(1,labels=xtlab[tpos],at=tpos, ...)
  axis(2, ...)
  if(with.legend){
    legend(legend.pos, legend=glabels,  lwd=lwd, lty=lty[1:k], col=col[1:k], horiz=horiz, cex=cex.legend)
  }

  return(invisible(k))
}
