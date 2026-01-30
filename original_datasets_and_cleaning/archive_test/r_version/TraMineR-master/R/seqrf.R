## creating rf sequence object
seqrf <- function(seqdata, diss, k=NULL, sortv="mds", weights=NULL,
                  weighted=TRUE,
                  grp.meth = "prop", squared = FALSE, pow = NULL){

  if (is.null(weights) & weighted) weights <- attr(seqdata,"weights")
  if (!weighted) weights <- NULL
  ## computing sortv
  if (!is.null(sortv)) {
    if (length(sortv)==1){
        if (sortv %in% c("from.start", "from.end")) {
          end <- if (sortv=="from.end") { max(seqlength(seqdata)) } else { 1 }
          beg <- if (sortv=="from.end") { 1 } else { max(seqlength(seqdata)) }

          sortv <- do.call(order, as.data.frame(seqdata)[,end:beg])
          sortv <- order(sortv)
        } else if (sortv != "mds") {
            stop(call.=FALSE, "If of length one, sortv must be one of 'mds', 'from.start', and 'from.end'")
        }
    } else if (length(sortv)!=nrow(seqdata)) {
      stop(call.=FALSE, "sortv must contain one value for each row in the sequence object ")
    } else {
      if (is.factor(sortv)) { sortv <- as.integer(sortv) }
    }
  }
  rf <- dissrf(diss, k=k, sortv=sortv, weights=weights,
               grp.meth = grp.meth, squared = squared, pow = pow)
  seqtoplot <- seqdata[rf[["medoids"]],]
  attr(seqtoplot,"weights") <- rf[["heights"]]
  srf <- list(seqtoplot=seqtoplot,rf=rf)
  class(srf) <- c("seqrf",class(srf))
  if (inherits(rf,"dissrfprop")){
    class(srf) <- c("seqrfprop",class(srf))
  }
  else
    class(srf) <- c("seqrfcrisp",class(srf))

  return(srf)
}

######
print.seqrf <- function(x, ...){
  print(x[["seqtoplot"]], ...)
}

#####
summary.seqrf <- function(object, format="SPS", dist.idx = 1:10, ...){
  #limit <- max(seqlength(seqdss(x[["seqtoplot"]])))
  sry <- summary(object[["rf"]], dist.idx = dist.idx)
  meds <- suppressMessages(seqformat(object[["seqtoplot"]], to=format, compress=TRUE,
                                                 SPS.out = list(xfix = "", sdsep = "/")))
  mnames <- sub("\\..","",x=rownames(meds))
  ##print(id)
  medoids <- data.frame(index=sry[["medoids"]],names=mnames,medoids=meds)
  rownames(medoids)<-NULL
  sry[["medoids"]] <- medoids
  return(sry)
}
