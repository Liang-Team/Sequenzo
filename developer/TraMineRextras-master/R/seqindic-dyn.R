## Evolution of indicator
## Inspired from Pelletier et al., 2020

seqindic.dyn <- function(seqdata, indic="cplx", window.size = .2, sliding = TRUE,
      wstep=1, with.missing=FALSE, endmiss.as.void=FALSE, silent.indic = TRUE, ...) {

	if (!inherits(seqdata,"stslist"))
		TraMineR:::msg.stop("data is NOT a sequence object, see seqdef function to create one")

  if (length(indic) > 1) {
    if (indic[1]!="Develop") {
      TraMineR:::msg.warn("Vector indic, only first value is used!")
      indic <- iindic <- indic[1]
    } else if (length(indic) > 2){
      TraMineR:::msg.warn("Develop vector indic, only first two value are used!")
      indic <- indic[1:2]
      iiindic <- iindic[2]
    } else iindic <- indic[2]
  } else iindic <- indic
  group.list <- c("all","basic","diversity","complexity","binary","ranked")
  if (iindic %in% group.list)
		TraMineR:::msg.stop("Bad indic value, group name not supported!")

  wstep <- as.integer(wstep)
  if (wstep < 1)
		TraMineR:::msg.stop("wstep must be a strictly positive integer!")

  lgth <- seqlength(seqdata, with.missing=TRUE)
  maxl <- max(lgth)
  if (window.size <= 0)
    TraMineR:::msg.stop("bad window.size value!")
  if (window.size < 1)
    step <- ceiling(window.size * maxl)
  else {
    step <- min(window.size, maxl)
  }

  re <- step
  rs <- 1
  slid <- as.integer(sliding)

  nr <- nrow(seqdata)
  nc <- ceiling((maxl-step+1)/wstep)
  j <- 1

  if (sliding) {
    if (silent.indic){
      windic <- function(k, seqdata, indic, with.missing, ...){
        suppressMessages(seqindic(seqdata[,(k-step+1):k], indic=indic, with.missing=with.missing, ...))
      }
    } else {
      windic <- function(k, seqdata, indic, with.missing, ...){
        seqindic(seqdata[,(k-step+1):k], indic=indic, with.missing=with.missing, ...)
      }
    }
  } else {
    if (silent.indic){
      windic <- function(k, seqdata, indic, with.missing, ...){
        suppressMessages(seqindic(seqdata[,1:k], indic=indic, with.missing=with.missing, ...))
      }
    } else {
      windic <- function(k, seqdata, indic, with.missing, ...){
        seqindic(seqdata[,1:k], indic=indic, with.missing=with.missing, ...)
      }
    }
  }
  ##ind.dyn <- sapply(re:maxl, FUN=windic, seqdata=seqdata, indic=indic, with.missing=with.missing, ...)

  ##ind.dyn <- matrix(unlist(ind.dyn), ncol=maxl-re+1)


  ind.dyn <- matrix(NA, nrow=nr, ncol=nc)
  re.range <- seq(from=re, to=maxl, by=wstep)
  if (length(re.range) < 2)
    TraMineR:::msg.stop("There is only one window, wstep probably too large!")
  void <- attr(seqdata,"void")
  nr <- attr(seqdata,"nr")
  miss.code <- if (endmiss.as.void & !with.missing) c(void,nr) else void
  #miss.code <- if (with.missing) void else c(void,nr)
  j <- 0
  for (k in re.range) {
    j <- j+1
    ## k is position of end of window
    ind.dyn[,j] <- as.matrix(windic(k, seqdata=seqdata, indic=indic, with.missing=with.missing, ...))
    ## setting NA for uncomplete windows
    wna <- seqdata[,k] %in% miss.code
    ind.dyn[wna,j] <- NA
###  while (re < maxl + 1) {
###     ind.dyn[,j] <- seqindic(seqdata[,rs:re], indic=indic, with.missing=with.missing, ...)[,1]
###     j  <- j+1
###     rs <- rs+slid
###     re <- re+1
###   }
  }

  colnames(ind.dyn) <- colnames(seqdata)[re.range]
  rownames(ind.dyn) <- rownames(seqdata)


  class(ind.dyn) <- c("dynin",class(ind.dyn))
  attr(ind.dyn, "weights") <- attr(seqdata,"weights")
  attr(ind.dyn,"xtstep") <- attr(seqdata,"xtstep")
  attr(ind.dyn,"tick.last") <- attr(seqdata,"tick.last")
  attr(ind.dyn, "window.size") <- step
  attr(ind.dyn, "wstep") <- wstep
  attr(ind.dyn, "sliding") <- sliding
  attr(ind.dyn, "indic") <- iindic

  return(ind.dyn)
}

#################################

print.dynin <- function(x, ...){
  dimx <- dim(x)
  dimnamesx <- dimnames(x)
  attributes(x) <- NULL
  dim(x) <- dimx
  dimnames(x) <- dimnamesx
  print(x, ...)
}

#################################

plot.dynin <- function(x, fstat=weighted.mean, group=NULL, conf=FALSE,
     main="auto", col=NULL, lty=NULL, lwd=3.5, ylim=NULL,
     ylab=NULL, xlab=NULL, xtlab=NULL, xtstep=NULL, tick.last=NULL,
     with.legend=TRUE, glabels=NULL, legend.pos="topright",
     horiz=FALSE, cex.legend=1, bcol=NULL, na.rm=FALSE,
     ret=FALSE, ...){

  if (!inherits(fstat,"function")) TraMineR:::msg.stop("fstat must be a function!")

  is.w.mean <- identical(fstat,weighted.mean)
  is.mean <- identical(fstat,mean)

  p.conf <- 0.975
  if (!is.logical(conf)){
    if (!is.numeric(conf) | conf <= 0 | conf >= 1)
      TraMineR:::msg.stop("Invalid conf value: should be logical or real between 0 and 1.")
    p.conf <- 1-(1-conf)/2
    conf <- TRUE
  }
  if (conf & !is.mean & !is.w.mean){
      TraMineR:::msg.warn("conf=TRUE works only with fstat=mean or weighted.mean, conf set as FALSE")
      conf=FALSE
  }

  if (is.mean | is.w.mean) fstat <- function(x){mean(x, na.rm=na.rm)}
  if (is.w.mean) {
    wt <- attr(x,"weights")
    if (is.null(wt)) {
      #fstat <- function(x,na.rm)mean(x, na.rm=na.rm)
      is.w.mean <- FALSE
    }
  }

  nc <- ncol(x)
  nr <- nrow(x)

  if (is.null(group)){
    group <- rep(1,nr)
  }
  if (!is.factor(group)) group <- factor(group)
  ngrp <- length(levels(group))


  tab.grp <- matrix(NA, nrow=ngrp, ncol=nc)

  #for(i in 1:nc) {
  #  tab.grp[,i] <- tapply(x[,i],group,fstat)
  #}

  if (is.w.mean){

    fsapp <- function(y,group,wt,na.rm){
      sapply(levels(group),
        function(whichpart) weighted.mean(x = y[group == whichpart],
                                          w = wt[group == whichpart],
                                          na.rm = na.rm))
    }
    tab.grp <- apply(x,2,fsapp,group,wt,na.rm)

    if (conf) {
      n.grp <- tapply(wt,group,sum)
      meanx2 <- apply(x^2,2,fsapp,group,wt,na.rm)
      sdev <- sqrt(meanx2 - tab.grp^2)
      err.grp <- qnorm(p.conf)*sdev
      err.grp <- err.grp/as.vector(sqrt(n.grp))
      U.grp <- tab.grp + err.grp
      L.grp <- tab.grp - err.grp
      dim(U.grp) <- dim(L.grp) <- c(ngrp,nc)
    }
  }
  else
  {
    tab.grp <- apply(x,2,tapply,group,fstat)

    if (conf) {
      n.grp <- tapply(rep(1,length(group)),group,sum,na.rm=na.rm)
      err.grp <- qnorm(p.conf)*apply(x,2,tapply,group,sd,na.rm=na.rm)
      err.grp <- err.grp/as.vector(sqrt(n.grp))
      U.grp <- tab.grp + err.grp
      L.grp <- tab.grp - err.grp
      dim(U.grp) <- dim(L.grp) <- c(ngrp,nc)
    }
  }

  dim(tab.grp) <- c(ngrp,nc)

  colnames(tab.grp) <- colnames(x)
  rownames(tab.grp) <- levels(group)


  tab.grp.ori <- tab.grp ## will return table before transformation
  na.rows <- apply(tab.grp,1,function(x)all(is.na(x)))
  rows.with.na <- apply(tab.grp,1,function(x)any(is.na(x)))
  #if (sum(rows.with.na) > sum(na.rows)) {
  #    TraMineR:::msg.warn("Rows of summary table with some NA turned into full NA rows!")
  #    tab.grp[rows.with.na,] <- NA
  #}
  #non.na.rows <- !rows.with.na
  if (all(is.na(tab.grp))){
      TraMineR:::msg.warn("Only NA in summary table, nothing to plot.")
      return(tab.grp)
  }

  #default.col <- brewer.pal(9,"Set1")
  default.col <- qualitative_hcl(ngrp, palette = "Dark 3")
  ##default.col <- c("red","blue","black","magenta","green")

  if(is.null(col)) {
     #col <- colors.list[1:k]
     col <- default.col
  }
  kk <- ceiling(ngrp/length(col))
  col <- rep(col,kk)
  col <- col[1:ngrp]

  if (conf) {
    default.bcol <- qualitative_hcl(ngrp, palette = "Pastel 1")
    if(is.null(bcol)) {
       #col <- colors.list[1:k]
       bcol <- default.bcol
    }
    kk <- ceiling(ngrp/length(bcol))
    bcol <- rep(bcol,kk)
    bcol <- bcol[1:ngrp]
  }

  default.lty <- c("solid","dashed","dotted")
  if(is.null(lty)) {
     lty <- default.lty
  }
  kk <- ceiling(ngrp/length(lty))
  lty <- rep(lty,kk)
  lty <- lty[1:ngrp]

  kk <- ceiling(ngrp/length(lwd))
  lwd <- rep(lwd,kk)
  lwd <- lwd[1:ngrp]

  if(is.null(glabels)) glabels <- levels(group)
  if(is.null(xtstep)) xtstep <- attr(x,"xtstep")
  if(is.null(tick.last)) tick.last <- attr(x,"tick.last")
  if(is.null(xtlab)) xtlab <- colnames(x)
  if(is.null(xlab)){
    slid.text <- ifelse(attr(x,"sliding"), "sliding window (", "incremental window (start ")
    xlab<-paste0("End of ",slid.text,"win size: ", attr(x,"window.size"),")")
  }
  if(is.null(ylab)) ylab<-attr(x,"indic")
  if(!is.null(main) && main[1] == "auto")
    main=paste("Dynamic index",attr(x,'indic'))

  if(is.null(ylim)){
        maxe <- max(tab.grp, na.rm=TRUE)
        mine <- min(tab.grp, na.rm=TRUE)
        if (mine==maxe) maxe <- mine + .1
        ylim <- c(floor(10*mine),ceiling(10*maxe))/10
  }

  ## frame
  plot(0, type= "n", axes=FALSE, xlab=xlab, ylab=ylab, main=main, ylim=ylim, xlim=c(1,nc), ...)
	tpos <- seq(from=1, to=nc, by=xtstep)
  if (tick.last & tpos[length(tpos)] < nc) tpos <- c(tpos,nc)
  axis(1,labels=xtlab[tpos], at=tpos, ...)
  #axis(1)
  axis(2, ...)

  if (conf) {
      #set.seed(1234)
      #df <- data.frame(x =1:10,
      #           F =runif(10,1,2),
      #           L =runif(10,0,1),
      #           U =runif(10,2,3))


      #plot(df$x, df$F, ylim = c(0,4), type = "l")
      #make polygon where coordinates start with lower limit and
      # then upper limit in reverse order
      for (i in 1:ngrp) {
        L.grp[i,is.na(L.grp[i,])] <- tab.grp[i,is.na(L.grp[i,])]
        U.grp[i,is.na(U.grp[i,])] <- tab.grp[i,is.na(U.grp[i,])]
        tab.na <- which(is.na(tab.grp[i,]))
        tabg <- tab.grp[i,]
        if (length(tab.na) > 0) {
          for (j in 1:length(tab.na)) {
            jj <- tab.na[j]
            if (jj > 1)
              L.grp[i,jj] <- U.grp[i,jj] <- tabg[jj] <- tabg[jj-1]
            else {
              TraMineR:::msg.warn("Some confidence bands not drawn because of NA at first position in summary table!!")
              break
            }
          }
        }


        poly.range <- 1:nc
        if (length(tab.na) > 0) poly.range <- 1:(min(tab.na)-1)
        if (poly.range[length(poly.range)] > 1) {
          polygon(c(poly.range,rev(poly.range)),c(L.grp[i,poly.range],rev(U.grp[i,poly.range])), col = bcol[i], border = FALSE)
          #add lines on borders of polygon
          lines(L.grp[i,poly.range], col=col[i],  type="l", lty=2)
          lines(U.grp[i,poly.range], col=col[i],  type="l", lty=2)
        }
      }
  }

  for (i in 1:ngrp) {
    if (any(!is.na(tab.grp[i,])))
     lines(tab.grp[i,], col=col[i],  type="l", lty=lty[i], lwd=lwd[i], ...)
  }


  ## legend
  non.na.rows <- apply(tab.grp,1,function(x)any(!is.na(x)))
  if(with.legend & ngrp>1){
    legend(legend.pos, legend=glabels[non.na.rows],
      lwd=lwd[non.na.rows], lty=lty[non.na.rows], col=col[non.na.rows],
      horiz=horiz, cex=cex.legend)
  }


  if (ret) {
    if (conf) {
      attr(tab.grp.ori,"L.grp") <- L.grp
      attr(tab.grp.ori,"U.grp") <- U.grp
    }
    return(tab.grp.ori)
  }
}
