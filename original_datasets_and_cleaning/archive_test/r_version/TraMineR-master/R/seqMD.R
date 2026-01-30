## multidomain sequences:
## - MD sequences of combined states and expanded alphabet,
## - additive trick costs (CAT),
## - OM distances based on CAT costs

## alias for backward compatibility
seqdistmc <- function(channels, what="diss", ch.sep="@@@@TraMineRSep@@@@", ...){
    seqMD(channels = channels, what=what, ch.sep=ch.sep, ...)
}

seqMD <- function(channels, method=NULL, norm="none", indel="auto", sm=NULL,
	with.missing=NULL, full.matrix=TRUE, link="sum", cval=2, miss.cost=2, cweight=NULL,
  what="MDseq", ch.sep = "+", fill.with.miss = TRUE) {

	## Checking arguments
  if (what=="sm") {
    what <- "cost"
    msg.warn("what='sm' deprecated! Use what='cost' instead.")
  }
  if (what=="seqmc") {
    what <- "MDseq"
    msg.warn("what='seqmc' deprecated! Use what='MDseq' instead.")
  }
  whatlist <- c("MDseq","cost","diss")
  if (!(what %in% whatlist)){
    msg.stop("what should be one of ",paste0("'",whatlist,"'", collapse=","))
  }

  if (what=="diss" && (is.null(method) || is.na(method)))
    msg.stop("a valid method must be provided when what = 'diss'")
  if (what=="cost" & is.null(sm))
    msg.stop("sm cannot be NULL when what = 'cost'")
  ## if time varying sm are provided, all sm must be 3-dimensional
  if (any(length(dim(sm[[1]]))==3) && !all(length(dim(sm[[1]]))==3))
    msg.stop("One sm is 3-dimensional and some are not!")
  timeVarying <- length(dim(sm[[1]])) == 3

  if(length(indel) > 1 & any(indel=="auto"))
    stop(" [!] 'auto' not allowed in vector or list indel")
  if(is.list(indel) & length(indel)==1)
    stop(" [!] When a list, indel must be of length equal to number of domains")

	nchannels <- length(channels)
	if (nchannels < 2) {
		stop("[!] please specify at least two domains")
	}
	if (is.null(cweight)) {
		cweight <- rep(1, nchannels)
	}

    has.miss <- rep(FALSE,nchannels)
  if (length(with.missing)==1) {
      with.missing <- rep(with.missing, nchannels)
  }
  for (i in 1:nchannels){
    if (length(grep(ch.sep, alphabet(channels[[i]], with.missing=TRUE), fixed=TRUE))>0)
      stop(" [!] ch.sep symbol (",ch.sep,") occurs in alphabet of at least one channel")
    has.miss[i] <- any(channels[[i]]==attr(channels[[i]],"nr"))
    if (!is.null(with.missing) && has.miss[i] != with.missing[i]) {
      msg.warn(paste(" Bad with.missing value for domain",i,". I set it as",has.miss[i]))
      ##with.missing[i]=has.miss[i]
    }
  }
  if (is.null(with.missing)) with.missing <- has.miss
  if (is.list(indel) & length(indel) != nchannels)
		stop("[!] when a list, indel must be of length equal to number of domains")
  if (length(with.missing) > 1 & length(with.missing) != nchannels )
		stop("[!] when a vector, with.missing must be of length equal to number of domains")

	numseq <- sapply(channels,nrow)
	if(any(numseq!=numseq[1])) {
		stop(" [!] sequence objects have different numbers of rows")
	}
	numseq <- numseq[1]
	msg.warn(nchannels, " domains with ", numseq, " sequences")
	## Actually LCP and RLCP are not included

  if (what=="diss") {
  	metlist <- c("OM", "LCS", "DHD", "HAM")
  	if (!method %in% metlist) {
  		stop(" [!] method must be one of: ", paste(metlist, collapse=" "), call.=FALSE)
  	}
  	## We handle LCS as OM
  	if (method=="LCS") {
  		method <- "OM"
  		sm <- "CONSTANT"
  		indel <- 1
  		cval <- 2
  		miss.cost <- 2
  	}
	  timeVarying <- method %in% c("DHD")
  	## Indels and sm only apply for OM
  	## Correct number of arguments
  	## Generating default substitution arguments for DHD ## not HAM
  	if (is.null(sm)) {
  		costmethod <- "CONSTANT"
  		if (method == "DHD") {
  			costmethod <- "TRATE"
  		}
  		sm <- rep(costmethod, nchannels)
  	}
  }
  if (length(sm)==1 && sm %in% c("CONSTANT", "TRATE", "INDELS", "INDELSLOG")){
 	sm <- rep(sm, nchannels)
  }
  if (length(indel)==1) {
   	indel <- rep(indel, nchannels)
  }
	## Checking correct numbers of info per channel
  if (what != "MDseq") {
	if ((length(indel)!= nchannels) ||
		(length(sm)!= nchannels) ||
		(length(cweight)!= nchannels)) {
		  stop(" [!] you must supply one weight, one substitution matrix, and one indel per domain")
    }
	}
	## indels
	if (is.list(indel))
    indel_list <- list()
  else
    indel_list <- numeric(length=nchannels)
  if (any(indel == "auto") & any(sm %in% c("INDELSLOG","INDELS")))
    indel_list <- list()
	## subsitution matrix
	substmat_list <- list()
	## alphabet for each channel
	alphabet_list <- list()
	## alphabet size per channel
	alphsize_list <-list()
	## seqlenth of each channels
	maxlength_list <- numeric(length=nchannels)
  ## Storing number of columns and cnames
  for (i in 1:nchannels)
  	maxlength_list[i] <- ncol(channels[[i]])
  md.cnames <- colnames(channels[[which.max(maxlength_list)]])

  ######################


	## Checking that all channels have the same length
	slength1 <- seqlength(channels[[1]])
	for (i in 2:nchannels) {
		if (sum(slength1 != seqlength(channels[[i]]))>0) {
			#if (with.missing) {
			#	stop(" [!] Some individuals have channels of different length. Set 'with.missing=TRUE' for all channels.")
			#} else {
				msg.warn("Cases with sequences of different length across domains!")
				break
			#}
		}
	}
	## ================================
	## Building the new sequence object
	## ================================
	message(" [>] building MD sequences of combined states...", appendLF=F)
	## Complex separator to ensure (hahem) unicity
	##sep <- "@@@@TraMineRSep@@@@"  ## now argument ch.sep
  sep <- ch.sep
	maxlength=max(maxlength_list)
	newseqdata <- matrix("", nrow=numseq, ncol=maxlength)
    rownames(newseqdata) <- rownames(channels[[1]])
	newseqdataNA <- matrix(TRUE, nrow=numseq, ncol=maxlength)
	for (i in 1:nchannels) {
		seqchan <- channels[[i]]
		void <- attr(seqchan, "void")
		nr <- attr(seqchan, "nr")
		for (j in 1:maxlength) {
			## No column in stslist object, filling with voids
			if (j > maxlength_list[i]) {
				newCol <- as.character(rep(void, numseq))
			}
			else {
				newCol <- as.character(seqchan[,j])
			    #}
    			## If fill.with.miss==TRUES, set void as nr
                if (fill.with.miss==TRUE & has.miss[i] & any(newCol==void)) {
        		     newCol[newCol == void] <- nr
                }
			    ## If void in all channels, then combined state will be void
    			newseqdataNA[,j] <- newseqdataNA[,j] & newCol == void
            }
			if (i > 1) {
				newseqdata[,j] <- paste(newseqdata[,j], newCol, sep = sep)
			}
			else {
				newseqdata[,j] <- newCol
			}
		}
  }
	## Setting void states back to NA  (nr will be considered as a distinct state)
  newseqdata[newseqdataNA] <- NA

  ## since v 2.2-0 automatic cpal no longer limited to 12 states, so no need of the following
	#alphabet_size <- length(unique(as.character(newseqdata))) - as.integer(sum(is.na(newseqdata))>0)
	#suppressMessages(newseqdata <- seqdef(newseqdata, cpal=rep("blue", alphabet_size)))
  suppressMessages(newseqdata <- seqdef(newseqdata, cnames=md.cnames))
  message(" OK")

  if (what == "MDseq") {
    return(newseqdata)
  }
  else { ## what == "diss" or "cost"
  ######################
  	## ============================================================
  	## Building and checking substitution matrix per channel
  	## ============================================================
  	for (i in 1:nchannels) {
  		## Sequence object
  		if (!inherits(channels[[i]],"stslist")) {
  			stop(" [!] channel ", i, " is not a state sequence object, use 'seqdef' function to create one", call.=FALSE)
  		}
  		alphabet_list[[i]] <- attr(channels[[i]],"alphabet")
  		## Checking missing values
      ##cat("i = ",i,"with.missing = ",with.missing,"\n")
  		if (with.missing[i]) {
  			alphabet_list[[i]] <- c(alphabet_list[[i]],attr(channels[[i]],"nr"))
  			message(" [>] including missing value as an additional state" )
  		}
  		else {
  			if (any(channels[[i]]==attr(channels[[i]],"nr"))) {
  				stop(" [!] found missing values in channel ", i, ", set with.missing as TRUE for that channel")
  			}
  		}
  		alphsize_list[[i]] <- length(alphabet_list[[i]])
      if(is.list(indel)){
        if (length(indel[[i]])==1)
          indel[[i]] <- rep(indel[[i]],alphsize_list[[i]])
        if (length(indel[[i]]) != alphsize_list[[i]]){
          cat("domain = ",i,", indel length = ", length(indel[[i]]), ", alphabet size = ", alphsize_list[[i]], "\n alphabet = ", alphabet_list[[i]],"\n" )
  				stop(" [!] indel length does not match size of alphabet for at least one channel")
        }
      }
      else if (!any(indel=="auto") & !is.list(indel_list)) {
  		  indel_list[i] <- indel[i]
      }

  		## Substitution matrix generation method is given
  		if	(is.character(sm[[i]])) {
  			message(" [>] computing substitution cost matrix for domain ", i)
  			costs <- suppressMessages(seqcost(channels[[i]], sm[[i]], with.missing=has.miss[i],
  				time.varying=timeVarying, cval=cval, miss.cost=miss.cost))
            substmat_list[[i]] <- costs$sm
            if (any(indel=="auto")) {
              if (is.list(indel_list)){
                if (length(costs$indel)==1) costs$indel <- rep(costs$indel,alphsize_list[[i]])
                indel_list[[i]] <- costs$indel
              }
              else
                indel_list[i] <- costs$indel
            }
  		}
  		else { ## provided sm
        #message(" [>]   channel ",i)

            if (any(indel[i] == "auto") & !is.list(indel_list))
              indel_list[i] <- max(sm[[i]])/2
            else
              indel_list[i] <- indel[i]
              #cat("\n indel_list[i] ",indel_list[[i]], "\n with.missing = ",with.missing,"\n")
              #print(sm[[i]])
   			substmat_list[[i]] <- sm[[i]]
        }
  		## Checking correct dimension of indel and cost matrix
   		checkcost(substmat_list[[i]], channels[[i]], with.missing = has.miss[i], indel = indel_list[[i]])

  		## Mutliply by channel weight
  		substmat_list[[i]] <- cweight[i]* substmat_list[[i]]
  	}
    if (any(indel=="auto")) indel <- indel_list
  #}


  	## =========================================
  	## Building the new CAT substitution cost matrix
  	## =========================================
  	message(" [>] computing MD substitution and indel costs with additive trick...", appendLF=FALSE)
  	## Build subsitution matrix and new alphabet
  	alphabet <- attr(newseqdata,"alphabet")
  	alphabet_size <- length(alphabet)
    newindel <- NULL
  	## Recomputing the substitution matrix
  	if (!timeVarying) {
  		newsm <- matrix(0, nrow=alphabet_size, ncol=alphabet_size)
      if (is.list(indel)){
        newindel <- rep(0,alphabet_size)
        statelisti <- strsplit(alphabet[alphabet_size], sep, fixed=TRUE)[[1]]
        ##cat("\n last state statelisti ",statelisti, "\n")
        for (chan in 1:nchannels){
		  ipos <- match(statelisti[chan], alphabet_list[[chan]])
          newindel[alphabet_size] <- newindel[alphabet_size] + indel[[chan]][ipos]*cweight[chan]
           ##cat("ipos ",ipos,"\n indel chan ",chan,": ",indel[[chan]],"\n newindel = ",newindel,"\n" )
        }
      }
  		for (i in 1:(alphabet_size-1)) {
  			statelisti <- strsplit(alphabet[i], sep, fixed=TRUE)[[1]]
        ##cat("state ",i," statelisti ",statelisti, "\n")
            if (is.list(indel)){
                for (chan in 1:nchannels){
  					 ipos <- match(statelisti[chan], alphabet_list[[chan]])
                newindel[i] <- newindel[i] + indel[[chan]][ipos]*cweight[chan]
           ##cat("ipos ",ipos,"\n indel chan ",chan,": ",indel[[chan]],"\n newindel = ",newindel,"\n" )
                }
            }
  			for (j in (i+1):alphabet_size) {
  				cost <- 0
  				statelistj <- strsplit(alphabet[j], sep, fixed=TRUE)[[1]]
  				for (chan in 1:nchannels) {
  					ipos <- match(statelisti[chan], alphabet_list[[chan]])
  					jpos <- match(statelistj[chan], alphabet_list[[chan]])
  					cost <- cost + substmat_list[[chan]][ipos, jpos]
  				}
  				newsm[i, j] <- cost
  				newsm[j, i] <- cost
  			}
  		}
  	} else {
  		## Recomputing time varying substitution
  		newsm <- array(0, dim=c(alphabet_size, alphabet_size, maxlength))
  		for (t in 1:maxlength) {
  			for (i in 1:(alphabet_size-1)) {
  				statelisti <- strsplit(alphabet[i], sep, fixed=TRUE)[[1]]
  				for (j in (i+1):alphabet_size) {
  					cost <- 0
  					statelistj <- strsplit(alphabet[j], sep, fixed=TRUE)[[1]]
  					for (chan in 1:nchannels) {
  						ipos <- match(statelisti[chan], alphabet_list[[chan]])
  						jpos <- match(statelistj[chan], alphabet_list[[chan]])
  						cost <- cost + substmat_list[[chan]][ipos, jpos, t]
  					}
  					newsm[i, j, t] <- cost
  					newsm[j, i, t] <- cost
  				}
  			}
  		}
  	}
    rownames(newsm) <- colnames(newsm) <- alphabet  ## labels are too long
  	message(" OK")
  	## Indel as sum
    if (is.null(newindel) & !is.list(indel_list)) {
  	   newindel <- sum(indel*cweight)
    }
  	## If we want the mean of cost..
  	if (link=="mean") {
  		newindel <- newindel / sum(cweight)
  		newsm <- newsm / sum(cweight)
  	}
    if (what == "cost") {
      attr(newsm,"indel") <- newindel
      attr(newsm,"alphabet") <- alphabet
      attr(newsm,"cweight") <- cweight
      return(newsm)
    }
  }
  if (what == "diss") {
       if (any(is.na(newsm)) || any(is.na(newindel)))
            msg.stop("NA indel and/or substitution CAT costs, cannot compute MD distances!")
	   message(" [>] computing MD distances using additive trick ...")
      ##cat(" newindel = ",newindel,"\n")
	   ## Calling seqdist
	   return(seqdist(newseqdata, method=method, norm=norm, indel=newindel,
		        sm=newsm, with.missing=FALSE, full.matrix=full.matrix))
  }
}
