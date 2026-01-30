seqivardur <- function(seqdata, type=1, with.missing=FALSE){

	if (!inherits(seqdata,"stslist"))
		stop(" [!] data is NOT a sequence object, see seqdef function to create one")
  if (!type %in% c(1,2))
		stop(" [!] type should be 1 or 2!")

	dss <- suppressMessages(seqdss(seqdata, with.missing=with.missing))

  lgth <- seqlength(seqdata, with.missing=with.missing)
  dlgth <- suppressMessages(seqlength(dss, with.missing=with.missing))
  sdist <- suppressMessages(seqistatd(seqdata, with.missing=with.missing))
  nnvisit <- rowSums(sdist==0)

  ## the var function in R gives the unbiased variance with the n-1 denominator
  ## but we need the real variance here
  realvar <- function(x) {
  	n <- sum(!is.na(x))
  	var <- 1/n*sum((x - mean(x,na.rm=TRUE))^2,na.rm=TRUE)
  	return(var)
  	}

	dur <- seqdur(seqdata, with.missing=with.missing)

  if (type == 1) {
    ret <- apply(dur,1,realvar)
    meand <- apply(dur,1,mean,na.rm=TRUE)
    var.max <- (dlgth-1)*(1-meand)^2
  } else if (type == 2) {
    meand <- rowSums(dur, na.rm=TRUE)/(dlgth + nnvisit)
    ddur <- (dur - as.vector(meand))^2
    ret <- (rowSums(ddur, na.rm=TRUE) + nnvisit*meand^2) /  (dlgth + nnvisit)
    alph <- alphabet(seqdata, with.missing=with.missing)
    alph.size <- length(alph)
    if (alph.size<2) maxnnv <- 0
    else
      maxnnv <- ifelse(dlgth==1, alph.size-1, alph.size-2)
    meand.max <- meand*(dlgth + nnvisit)/(dlgth + maxnnv)
    ##print(meand.max)
    var.max <- ((dlgth-1)*(1-meand.max)^2 + (lgth-dlgth+1-meand.max)^2 + maxnnv*meand.max^2) / (dlgth + maxnnv)
  }

  ret <- as.vector(ret)
  attr(ret,"vmax") <- as.vector(var.max)
  attr(ret,"meand") <- as.vector(meand)

	class(ret) <- c("seqivardur", class(ret))
  return(ret)
}

print.seqivardur <- function(x, stat='var', ...) {
	## Conversion for printing without attributes
  statlist <- c('mean','std','var','vmax','all')
  if (any(!stat %in% statlist)) msg.stop.in("stat",statlist)

  dvar <- as.vector(x)
  std <- sqrt(as.vector(x))
  meand <- as.vector(attr(x,'meand'))
  vmax <- as.vector(attr(x,'vmax'))

  x <- cbind(meand,std,dvar,vmax)
  a <- which(stat=='all')
  if (length(a)>0)  stat <- stat[-a]
  if (length(stat) >0) x <- x[,which(statlist %in% stat)]

	NextMethod("print", ...)
}
