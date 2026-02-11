seqgranularity <- function(seqdata, tspan=3, method="last"){

  if (tspan == 1) return(seqdata)
  if (tspan < 1) stop("tspan must greater than 2")

  metlist <- c("first","first.valid","last","last.valid","mostfreq")
  if (!method %in% metlist) {
      stop(" [!] method must be one of: ", paste(metlist, collapse = " "),
          call. = FALSE)
  }


  n <- nrow(seqdata)
  missing.char <- attr(seqdata,"missing")
  nr <- attr(seqdata,"nr")
  void <- attr(seqdata, "void")
  is.void <- any(seqdata==void)
  lgth <- max(seqlength(seqdata))
  alph <- alphabet(seqdata)

  new.lgth <- ceiling(lgth/tspan)
  new.lgth.f <- floor(lgth/tspan)

  newseq <- seqdata[,1:new.lgth]
  newseq <- matrix(missing.char,nrow(seqdata),new.lgth)
  cnames <- names(seqdata)
  newcnames <- cnames[seq(from=1, to=lgth, by=tspan)]

  firstvalid <- function (seq, alphabet, first=TRUE, missing.char=NA) {
  	pos <- which(seq %in% alphabet)
  	if (length(pos)==0) res <- missing.char
  	else {
      if (first) fpos <- min(pos)
      else fpos <- max(pos)
      res <- seq[fpos]
    }
  	return(res)
	}

  prev <- 0
  if (method=="first") prev <- tspan - 1
  for (i in 1:new.lgth.f){
      newseq[,i] <- as.matrix(seqdata[,tspan*i - prev])
  }

  if (method %in% c("first.valid","last.valid")){
    first <- method == "first.valid"
    for (i in 1:new.lgth.f){
      seq <- seqdata[,(tspan*(i-1) + 1):(tspan*i)]
      newseq[,i] <- apply(seq, 1, firstvalid, alphabet=alph, first=first, missing.char=missing.char)
    }
  }
  if (method=="mostfreq") {
      for(i in 1:new.lgth.f) {
          st.freq <- suppressMessages(seqistatd(seqdata[,(tspan*(i-1) + 1):(tspan*i)]))
          newseq[,i] <- apply(st.freq,1,function(x){ifelse(max(x)==0,missing.char,names(which.max(x)))})
      }
  }
  if (new.lgth > new.lgth.f){
    if(lgth > tspan*new.lgth.f + 1) {
      if (method=="first"){
          newseq[,new.lgth] <- as.matrix(seqdata[,tspan*new.lgth.f + 1])
      } else if (method %in% c("first.valid","last.valid")){
          first <- method == "first.valid"
          seq <- seqdata[,(tspan*new.lgth.f+1):lgth]
          newseq[,new.lgth] <- apply(seq, 1, firstvalid, alphabet=alph, first=first, missing.char=missing.char)
      } else if (method=="mostfreq") {
          st.freq <- suppressMessages(seqistatd(seqdata[,(tspan*new.lgth.f+1):lgth]))
          newseq[,new.lgth] <- apply(st.freq,1,function(x){ifelse(max(x)==0,missing.char,names(which.max(x)))})
      } else { # method=="last"
          newseq[,new.lgth] <- as.matrix(seqdata[,lgth])
      }
    }
    else newseq[,new.lgth] <- as.matrix(seqdata[,lgth])
    newcnames[new.lgth] <- cnames[lgth]
  }
  newseq <- as.matrix(newseq)
  newseq[newseq==nr] <- missing.char
  newseq[newseq==void] <- missing.char

  newseq <- suppressMessages(seqdef(newseq,
              alphabet=alph,
              weights =attr(seqdata,"weights"),
              labels  =attr(seqdata,"labels"),
              start   =attr(seqdata,"start"),
              missing =attr(seqdata,"missing"),
              nr      =attr(seqdata,"nr"),
              void    =attr(seqdata,"void"),
              #xtstep  =attr(seqdata,"xtstep"),
              cpal    =attr(seqdata,"cpal"),
              tick.last=attr(seqdata,"tick.last"),
              Version =attr(seqdata,"Version"),
              right   =ifelse(is.void,"DEL",NA)
              ))
  #}
  colnames(newseq) <- newcnames
  attr(newseq,"xtstep") <- ceiling(attr(seqdata,"xtstep")/tspan)
	
  return(newseq)
}
