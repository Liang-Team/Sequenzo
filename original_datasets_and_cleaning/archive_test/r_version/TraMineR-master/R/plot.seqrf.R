## plot method for objects seqrf produced by seqrf.
## author: Gilbert Ritschard

plot.seqrf <- function(x, space=0, border=NA, which.plot="medoids", ylab=NA,
                    main=NULL, frame.plot=FALSE, info="all", skipar = FALSE, ...){

    dotargs <- list(...)
    if (!skipar){
        def.par <- par(no.readonly = TRUE) # save default, for resetting...
        on.exit(par(def.par))
    }
    plot.types <- c("both","medoids","diss.to.med")
    if (! which.plot %in% plot.types)
        stop(" which.plot must be one of ", paste(plot.types, collapse=", "))
    info.types <- c("all","stat","subtitle","none")
    if (is.null(ylab)) ylab <- NA

    xaxt <- "s"
    if (!is.null(dotargs[["xaxis"]]))
        if(!dotargs[["xaxis"]]) xaxt <- "n"
    xaxis <- xaxt == "s"
    yaxt <- "n"
    if (!is.null(dotargs[["yaxis"]]) & which.plot=="diss.to.med")
        if(dotargs[["yaxis"]]) yaxt <- "s"
    yaxis <- yaxt == "s"

  sep.ylab <- (isFALSE(dotargs[["yaxis"]]) && (is.null(ylab) || !is.na(ylab)))
  cex.lab <- par("cex.lab")
  if ("cex.lab" %in% names(list(...))) cex.lab <- list(...)[["cex.lab"]]


    if(!skipar & which.plot=="both"){
  	  ##opar <- par(mfrow=c(1,2), oma=c(3,(!is.na(ylab)*5),(!is.null(main))*3,0), mar=c(1, 1, 2, 0))
  	  if (info %in% c("all","stat"))
        par(oma=c(3,0,(!is.null(main))*3,.5))
      else
        par(oma=c(0,0,(!is.null(main))*3,.5))
      layout(matrix(c(1,2),ncol=2), widths=c(.6,.4))
    }

    if (info %in% c("all","subtitle")){
      titmed <- "Group medoids"
      titbxp <- "Distances to medoids"
      if (!is.null(main)) {
        if (which.plot=="medoids")
            titmed <- paste(main,titmed, sep=": ")
        else if (which.plot=="diss.to.med")
            titbxp <- paste(main, titbxp, sep=": ")
      }
    }
    else if (!is.null(main) & which.plot == "both")
            titmed <- titbxp <- NULL
    else
        titmed <- titbxp <- main

    if (!skipar){
      if (!is.na(ylab))
        par(mar=c(xaxis * 2.5, 4 , (info %in% c("all","subtitle")) * 2, .5))
      else if (!is.null(main) & info %in% c("none","stat"))
        par(mar=c(xaxis * 2.5, 2 + yaxis , 2, .5))
      else
        par(mar=c(xaxis * 2.5, 2 + yaxis , (info %in% c("all","subtitle")) * 2, .5))
     }

   if (sep.ylab) {
        sylab <- ylab
        ylab <- NA
   }

  if (which.plot %in% c("medoids","both")){
     plot(x[["seqtoplot"]], idxs = 0, space=space, border=border, ylab=ylab, main=titmed, ...)
     if (sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)
  }

  #if (!is.null(main) & which.plot == "diss.to.med")
  #      titbxp <- paste(main,titbxp, sep=": ")
  if (which.plot %in% c("diss.to.med","both")){
     heights <- x[["rf"]][["heights"]]
     at      <- x[["rf"]][["at"]]
     pars = list(boxwex = 0.8, staplewex = 0.5, outwex = 0.5, frame.plot=frame.plot)
     if (which.plot == "both" & !skipar){
        if (!is.null(main) & info %in% c("none","stat"))
            par(mar=c(xaxis * 2.5, 0, 2, .5))
        else
            par(mar=c(xaxis * 2.5, 0, (info %in% c("all","subtitle")) * 2, .5))
     }
     wtd.boxplot.tmr(x[["rf"]][["dist.list"]], x[["rf"]][["weights.list"]], horizontal=TRUE, width=heights,
        main=titbxp, pars=pars, yaxt=yaxt, xaxt=xaxt, frame.plot=frame.plot,
        ylim=range(unlist(x[["rf"]][["dist.list"]])), at=at, ylab=ylab, cex.lab=cex.lab)

     if (which.plot=="diss.to.med" & sep.ylab)
        title(ylab=sylab, line=1, cex.lab=cex.lab)
  }

  if (which.plot=="both") {

  	if(!is.null(main)) title(main=main, outer=TRUE)
  	if(info %in% c("all","stat"))title(sub=sprintf("Representation quality: R2=%0.2f and F=%0.2f",
        x[["rf"]][["R2"]], x[["rf"]][["Fstat"]]), outer=TRUE, line=2)
  }

}
