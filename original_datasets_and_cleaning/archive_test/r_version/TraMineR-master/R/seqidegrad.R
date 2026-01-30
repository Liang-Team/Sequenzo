## degradation index
##  seqidegrad superseeds previous seqprecorr function

seqidegrad <- function(seqdata,
    state.order=alphabet(seqdata, with.missing), state.equiv=NULL,
    stprec=NULL, with.missing=FALSE,
    penalized="BOTH", method = "RANK", weight.type="ADD",
    pow=1, border.effect=10){

  if (is.logical(pow)) {
    spell.integr <- pow
    pow <- as.numeric(pow)
  } else {
    spell.integr <- TRUE
  }
  degr <- seqidegrad.private(seqdata,
            state.order=state.order, state.equiv=state.equiv, stprec=stprec,
            penalized=penalized, method=method, weight.type=weight.type,
            pow=pow, spell.integr=spell.integr,
            with.missing=with.missing, border.effect=border.effect)
  ##degr <- (1+degr)/2
  colnames(degr)<-"Degrad"
  return(degr)
}

## for backward compatibility
seqprecorr <- function(...){
  msg.stop("seqprecorr is obsolete, use seqidegrad with spell.integr=FALSE and method='TRATEDSS' instead")
  #olist <- list(...)
  #if (!"method" %in% names(olist)) olist[["method"]] <- 'TRATEDSS'
  #if (!"spell.integr" %in% names(olist)) olist[["spell.integr"]] <- FALSE

  #do.call(seqidegrad, olist)
}

### seqprecorr <- function(seqdata, state.order=alphabet(seqdata, with.missing), state.equiv = NULL,
###       penalized="BOTH", method="TRATEDSS", weight.type="ADD",
###       stprec=NULL, with.missing=FALSE, border.effect=10, tr.type) {
###
###   TraMineR.check.depr.args(alist(method = tr.type))
###
###   precorr <- seqidegrad.private(seqdata,
###               state.order=state.order, state.equiv=state.equiv, stprec=stprec,
###               penalized=penalized, method=method, weight.type=weight.type,
###               with.missing=with.missing, border.effect=border.effect)
###   return(precorr)
### }


## =====================================
## Correction term
## =====================================

seqidegrad.private <- function(seqdata, spell.integr=TRUE, state.order=alphabet(seqdata, with.missing),
      state.equiv = NULL, stprec=NULL,
      penalized="BOTH", method="TRATEDSS", weight.type="ADD", pow=1,
      with.missing=FALSE, border.effect=10) {

	if (!inherits(seqdata,"stslist"))
		msg.stop("seqidegrad: seqdata is NOT a sequence object, see seqdef function to create one")

  if(!is.null(stprec)) {
    if(length(stprec) != length(alphabet(seqdata, with.missing)))
      msg.stop("seqidegrad: length of stprec does not match the size of the alphabet!")
  }
  #if(method %in% c("RANK","RANK+") || spell.integr){
    stprec <- suppressMessages(seqprecstart(seqdata, state.order=state.order,
                        state.equiv=state.equiv, stprec=stprec, with.missing=with.missing))
    state.order <- attr(stprec,"state.order")
    state.equiv <- attr(stprec,"state.equiv")
  #}


  if (is.logical(penalized)){
    if(penalized) penalized<-'BOTH' else penalized<-'NO'
  }
  if (penalized=="NO")
    return(0L)


  ## weight.type == "ADD"  : additive, i.e. 1-tr
  ##             == "INV"  : inverse, i.e. 1/tr
  ##             == "LOGINV"  : log inverse, i.e. log(1/tr)

  ## method == "FREQ"     : overall frequency of the transition
  ## method == "TRATEDSS" : transition prob in DSS sequences
  ## method == "TRATE"    : transition probabilities
  ## method == "RANK"    : diff of start costs
  ## method == "SPELLINT"    : spell integration index
  ## method == "ONE"

  ## penalized == "NEG" (default) negative transitions are penalized
  ##           == "POS" positive transition are negatively penalized
  ##           == "BOTH"
  ##           == "NO"  return a zero correction



  method.names <- c("FREQ","TRATE","TRATEDSS","RANK","FREQ+","TRATE+","TRATEDSS+","RANK+","ONE")
  if (!(method %in% method.names))
		msg.stop.in("method", method.names)

  use.mean.tr <- FALSE
  if (method %in% c("FREQ+","TRATE+","TRATEDSS+","RANK+")) {
    use.mean.tr <- TRUE
    method <- switch(method, "FREQ+"="FREQ","TRATE+"="TRATE","TRATEDSS+"="TRATEDSS","RANK+"="RANK")
  }

  if (method %in% c("FREQ","TRATE","TRATEDSS")){
    weight.names <- c('ADD','INV','LOGINV')
    if (!(weight.type %in% weight.names))
  		msg.stop.in("weight.type", weight.names)
    if (!(border.effect > 1))
  		msg.stop("border.effect should be strictly greater than one!")
  }

  pen.names <- c('NEG','POS','BOTH','NO')
  if (!(penalized %in% pen.names))
		msg.stop.in("penalized", pen.names)

  ## Checking that listed states are in alphabet
  state.order <- states.check(seqdata, state.order, state.equiv, with.missing)
	
	###################################
	## setting signs according to the rank order of the states

	alphabet <- alphabet(seqdata, with.missing)
  #if (with.missing) alphabet <- c(alphabet, attr(seqdata,"nr"))
  alph=length(alphabet)
  tr <- matrix(1, nrow=alph, ncol=alph)
  diag(tr) <- 0
  signs <- matrix(0, nrow=alph, ncol=alph)
	state.order.plus <- state.order
  state.noncomp <- NULL
	
  if (length(unique(state.order)) < alph){
    inoncomp <- which(is.na(match(alphabet,unique(state.order))))
    state.noncomp <- alphabet[inoncomp]
    message(" [>] Non ranked states: ", paste(state.noncomp, collapse=', '))
    state.order.plus <- c(state.order, state.noncomp)
  } else {
    state.order.plus <- state.order
  }

  ## To count an indirect transition through incomparable states as
  ## a transition from the first to the last state
  ## we replace incomparable states with the previous state
  ## No need to replace initial incomparable state
  ## since transitions from incomparable state will be ignored

  seqdata.ori <- seqdata ## just in case we would need the original later
  alphabet.ori <- alphabet

  ##print(seqdata)	
  ##ii <- as.matrix(which(seqdata=="I", arr.ind =TRUE))
  identnoncomp <- function(seqdata, noncomp){
    sm <- as.matrix(seqdata)
    ii <- which(sm %in% noncomp)
    im <- matrix(0,nrow=length(ii),ncol=2)
    im[,2] <- (ii-1) %/% nrow(sm) + 1
    im[,1] <- (ii-1) %% nrow(sm) + 1
    return(im)
  }

  ii <- identnoncomp(seqdata, state.noncomp)
  if(!is.matrix(ii)) ii <- as.matrix(t(ii)) ## when ii has only one row
  iii <- ii[ii[,2]>1,]
  if(!is.matrix(iii)) iii <- as.matrix(t(iii)) ## when iii has only one row
  continue <- TRUE
  nstep <- ncol(seqdata)
  step <- 1

  while(nrow(iii)>0 && step < nstep) {
    iin <- iii
    iin[,2] <- iin[,2] - 1
    #iin

    seqdata[iii] <- seqdata[iin]

    ##ii <- which(seqdata=="I", arr.ind =TRUE)
	  ii <- identnoncomp(seqdata, state.noncomp)
    if(!is.matrix(ii)) ii <- as.matrix(t(ii))
    iii <- ii[ii[,2]>1,]
    if(!is.matrix(iii)) iii <- as.matrix(t(iii))
    step <- step + 1
  }

  ## Using same state value for all elements in a same equivalence class
	if(!is.null(state.equiv)){
    s <- as.matrix(seqdata)
	  lc <- length(state.equiv)

	  for (i in 1:lc) {
      seqdata[matrix(s %in% state.equiv[[i]],nrow=nrow(s))] <- state.equiv[[i]][1]
    }
  }
	
  ## Number of transitions
  sl <- seqlength(seqdata, with.missing=with.missing)
  dss <- seqdss(seqdata, with.missing=with.missing)
  dssl <- seqlength(dss)
  nbseq <- nrow(dss)
  integr <- matrix(1, nrow=nbseq, ncol=max(dssl))


  ## Transition weights based on resulting spell integration index (Sept 2020)

  if (spell.integr) {
    Dur <- seqdur(seqdata, with.missing=with.missing)
    make.sps <- function(dur){
      sps <- paste0(1:length(dur),'/',dur)
      return(sps)
    }
    sps <- t(apply(Dur,1,make.sps))
    sps[is.na(Dur)] <- NA
    seqtmp <- suppressMessages(seqdef(sps, informat='SPS', SPS.in=list(xfix='',sdsep='/')))
    integr <- seqintegr(seqtmp, pow=pow)
  }

	##  default tr set above as 1s

  if (method %in% c('FREQ','TRATE','TRATEDSS')) {
	
    ## Computing transition probabilities

	  if (method == "FREQ") { ## default tr set above as 1s (no transition weight)
	    tr <- suppressMessages(seqtrate(dss, count=TRUE, with.missing=with.missing))  ## Here we compute the counts of the transitions
      tr <- tr / sum(tr) ## and now the proportion of each observed transition
	  }
	  else if (method == "TRATEDSS") {
	    tr <- suppressMessages(seqtrate(dss, with.missing=with.missing))
	  }
	  else if (method == "TRATE") {
	    tr <- suppressMessages(seqtrate(seqdata, with.missing=with.missing))
	  }

	  ## Computing transition weights from transition probabilities

    eps <- 1e-10
    ##border.effect <- 10
    diag(tr) <- 0
    ## adjustement when any tr close from 1
    if (any(tr > 1 - .1/border.effect)) tr <- tr - tr/border.effect

		if (weight.type == "ADD") {
		  tr <- 1 - tr
		}
		else if (weight.type == "INV"){
		  tr <- (1 + eps)/(tr + eps) ## - 1  ## GR 29/04/19 minus 1 to set min at 0
		}
		else if (weight.type == "LOGINV"){
		  tr <- log((1 + eps)/(tr + eps))
		}
  	##tr <- tr/tr[1,1] ## normalize by diagonal value
  	tr <- tr/diag(tr) ## normalize by diagonal value
  }
  else if (method == "RANK"){
      #for (j in 1:length(stprec))
      #  tr[,j] <- abs(stprec - stprec[j])
      tr <- matrix(rep(stprec,alph),alph,byrow=TRUE)
      tr <- abs(tr-stprec)
      rownames(tr) <- colnames(tr) <- alphabet ##(seqdata)
  }

	## reorder rows and colunms according to state.order
	
	ord <- match(state.order.plus,alphabet.ori)
	ordo <- match(alphabet.ori,state.order.plus)

  #tr <- tr[ord,ord]
	signs <- signs[ord,ord]	

	if (penalized=="NEG"){
	  signs[upper.tri(signs, diag = FALSE)] <- 1
	} else if (penalized=="POS"){
	  signs[lower.tri(signs, diag = FALSE)] <- -1
	} else if (penalized=='BOTH'){
	  signs[upper.tri(signs, diag = FALSE)] <- 1
	  signs[lower.tri(signs, diag = FALSE)] <- -1
	}

	## resetting original order
	#tr <- tr[ordo,ordo]
	signs <- signs[ordo,ordo]

  ## ignore transitions within equivalence classes
	if(!is.null(state.equiv)){
	  lc <- length(state.equiv)
	  for (i in 1:lc) {
	    iequiv <- match(state.equiv[[i]],alphabet)
	    signs[iequiv,iequiv] <- 0
	    tr[iequiv,iequiv] <- 0 ## useless, such transitions have been suppressed
      tr[,iequiv[-1]] <- 0  ## not used, just for clarity
      tr[iequiv[-1],] <- 0
	  }
}

	## ignore transitions to and from incomparable states
  ## affects only first transition in sequence starting with an incomparable state
	if (!is.null(state.noncomp)){
	  signs[,inoncomp] <- 0
	  signs[inoncomp,] <- 0
  	  tr[,inoncomp] <- 0  ## never used (check)
  	  tr[inoncomp,] <- 0
	}

  diag(tr) <- 0
  maxtr <- max(tr)
  	
	transw <- matrix(0, nrow=nbseq, ncol=1)
	rownames(transw) <- rownames(seqdata)
	transpen <- transw
	prop.transpen <- transw

  if (penalized != 'NO') {
	  dss.num <- seqasnum(dss, with.missing=with.missing)+1
	  ## sum of transition weights in each sequence
  	for (i in 1:nbseq) {
  		if (dssl[i]>1) {
  			for (j in 2:dssl[i]) {
          ## integr[i,j] = 1 when spell.integr=FALSE, default tr = 1
  			  transw[i] <- transw[i] + tr[dss.num[i,j-1], dss.num[i,j]] * integr[i,j]
  			  transpen[i] <- transpen[i] + tr[dss.num[i,j-1], dss.num[i,j]] * signs[dss.num[i,j-1], dss.num[i,j]] * integr[i,j]
  			}
  		}
  	  ## else leave prop.transpen[i] <- 0
  	}
    nz <- transw > 0
    if (spell.integr)
      prop.transpen[nz] <- transpen[nz]
    else
      prop.transpen[nz] <- transpen[nz]/transw[nz]

    if (use.mean.tr){
      mean.transw <- transw/dssl ## mean transition weight per sequence
      prop.transpen <- mean.transw * prop.transpen
    }
  }

	
	dimnames(signs) <- dimnames(tr)
	colnames(prop.transpen) <- "Penalty"
	attr(prop.transpen,"tr") <- tr
	attr(prop.transpen,"signs") <- signs
	attr(prop.transpen,"state.noncomp") <- state.noncomp
	attr(prop.transpen,"state.order") <- state.order.plus
	##attr(prop.transpen,"integr") <- integr
	##attr(prop.transpen,"norm.transpen") <- norm.transpen
	##attr(prop.transpen,"seqdata") <- seqdata
  class(prop.transpen) <- c("seqidegrad",class(prop.transpen))
	
	return(prop.transpen)
}

print.seqidegrad <- function(x, ...){
  names <- dimnames(x)
  attributes(x) <- NULL
  x <- as.matrix(x)
  dimnames(x) <- names
  print(x, ...)
}
