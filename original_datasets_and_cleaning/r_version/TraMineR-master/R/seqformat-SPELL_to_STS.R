# Should only be used through seqformat()

## ================================
## Convert from SPELL to STS format
## ================================

SPELL_to_STS <- function(seqdata, id=1, begin=2, end=3, status=4,
	process=TRUE, pdata=NULL, pvar=NULL,
	limit=100, overwrite=TRUE, fillblanks=NULL,
	tmin=NULL, tmax=NULL) {

	## if overwrite=TRUE, the latest spell overwrite the one before, if set to FALSE, the earlier is kept
	begincolumn <- seqdata[,begin]
	endcolumn <- seqdata[,end]

  if (inherits(seqdata[,begin], "Date") || is.character(begincolumn) || is.factor(begincolumn)
    || inherits(seqdata[,end], "Date") || is.character(endcolumn) || is.factor(endcolumn))
    stop(" [!] 'begin' and 'end' columns of seqdata should contain integer values", call.=FALSE)


  ##is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol
  is.wholenumber <- function(x){as.integer(x) == x}

	if (!all(is.wholenumber(c(endcolumn,begincolumn)), na.rm=TRUE)) {
		stop(" [!] Found non-integer values in colunms referred to by begin and/or end argument!", call.=FALSE)
	}
	if (any(begincolumn<1, na.rm=TRUE)) {
		stop(" [!] Found one or more spell with starting time < 1", call.=FALSE)
	}
	if (any(endcolumn-begincolumn<0, na.rm=TRUE)) {
		stop(" [!] Found one or more spell with ending time < starting time\n",
            paste(which(endcolumn-begincolumn<0), collapse=" "), call.=FALSE)
	}

  ## print("Testing whether process==FALSE and pdata=='auto'")
  if ( !process && !is.null(pdata) && !is.data.frame(pdata))  pdata <- NULL

	frmoption <- NULL

  ##print("Now testing pdata")
	if (is.data.frame(pdata)){
    if (inherits(pdata[,pvar[2]], "Date") || is.character(pdata[,pvar[2]]) || is.factor(pdata[,pvar[2]]))
      stop(" [!] Column '",pvar[2],"' of pdata should contain integer (birth year) values", call.=FALSE)
    #print("pdata ok")
	  pdata <- pdata[, pvar]
  }
	if (process==TRUE) {
		if (!is.null(pdata)) frmoption <- "year2age"
		else frmoption <- "age2age"
	}
  else { ## if(process==FALSE) {
		if (is.null(pdata)) frmoption <- "year2year"
		else frmoption <- "age2year"
	}

	## =========================
	## creation of the dataframe
	## =========================
	if(process==FALSE) {
		if(is.null(tmin) || is.null(tmax)) {
			if(frmoption=="year2year") {
				tmin <- min(begincolumn[!is.na(begincolumn) & begincolumn > 0])
				tmax <- max(endcolumn[!is.na(endcolumn) & endcolumn > 0])
			}
			if(frmoption=="age2year") {
				tmin <- min(pdata[,2][!is.na(pdata[,2]) & pdata[,2] > 0])
				max1 <- max(endcolumn[!is.na(endcolumn) & endcolumn > 0])
				max2 <- max(pdata[,2][!is.na(pdata[,2]) & pdata[,2] > 0])
				tmax <- max1 + max2
			}
			message(paste(" [>] time axis:", tmin, "->", tmax))
			if(is.na(tmin) || is.na(tmax)) {
				stop(" [!] Could not find the minimum or maximum year automatically, please use tmin/tmax options")
			}
		}
		limit <- (tmax - tmin) + 1
		year <- tmin
    #names.seqresult <- NULL
		#for(i in 1:limit) {
		#	names.seqresult <- c(names.seqresult, (paste("y", year, sep="")))
		#	year <- year+1
		#}
    names.seqresult <- paste0("y", seq(from = year, to = year+limit-1))
		##names.seqresult <- c(names.seqresult, "id")
	}
	else {
        if (length(endcolumn[!is.na(endcolumn) & endcolumn > 0]) > 0){
          maxend <- max(endcolumn[!is.na(endcolumn) & endcolumn > 0])
          ## Check of sequence length <= limit is done later during the transformation
    ###       if (frmoption == "year2age"){
    ###         #minbeg <- min(begincolumn, na.rm=TRUE)
    ###         #mlimit <- maxend - minbeg
    ###         #if (mlimit > limit)
    ###         #  msg.warn(paste("Some sequence lengths could be larger than 'limit' and will be truncated at",limit))
    ###       }
    ###       else if (maxend > limit) {
    ###         msg.warn(paste("max of 'end' column > limit! Sequences truncated at limit=",limit))
    ###       }
        } else {
          msg.warn("No positive value in 'end' column!")
        }
    		#names.seqresult <- c((paste("a", seq(1:limit), sep="")),"id")
    		names.seqresult <- paste0("a", seq(from = 1, to = limit))
	}

	#seqresult <- matrix(nrow=1, ncol=limit+1)
        # on récupère la liste des individus
	lid <- unique(seqdata[,id])
	#print(lid)
	nbseq <- length(lid)
  seqresult <- matrix(nrow=nbseq, ncol=limit)
	#seqresult <- as.data.frame(seqresult)
  status.orig <- seqdata[,status]
  ## if status is a factor, convert to integer for faster computation
	if (is.factor(seqdata[,status])) {
    seqdata[,status] <- as.integer(seqdata[,status])
        #	for (k in 1:(limit)) {
	#		seqresult[,k] <- factor(seqresult[,k], levels=levels(seqdata[,status]), labels=levels(seqdata[,status]))
	#	}
		if(!is.null(fillblanks)) {
			fillblanksF <- fillblanks
			fillblanks <- nlevels(status.orig)+1
		}
	}
  #  if (!is.null(fillblanks)) {

#		fillblanksv <- nlevels(status.orig)+1
#	}
	#names(seqresult) <- names.seqresult
	## ================================
	## end of creation of the dataframe
	## ================================


	#print(paste("nbseq = ", nbseq))
	# if birth year have been given in pdata dataframe, we retrieve ids and birth years
	birthyrid1<-0
	if((frmoption=="year2age" || frmoption=="age2year") && (!is.null(pdata) && any(pdata!="auto"))) {
		birthyrid1 <- pdata[,1]
		birthyr1 <- pdata[,2]
  }
	## ===============
	## individual loop
	## ===============
	#print(birthyrid1)
	#print(length(birthyrid1))
	#print(nbseq)
  #maxlength.warn <- TRUE
  nemptseq <- 0
  iemptseq <- NULL
  nstartbefore <- 0
  istartbefore <- NULL
  nendbefore <- 0
  iendbefore <- NULL
  ntrunc <- 0
  itrunc <- NULL
  ii <- 0
  #print(frmoption)
	for (i in 1:nbseq) {
		spell <- seqdata[seqdata[,id]==lid[i],]
		# number of spells for individual i
		idxmax <- nrow(spell)
		# we check if the first episode looks normal (starting age/year > 0)
    # (issues found with a file frome the Swiss panel)
		if(frmoption=="age2age" || frmoption=="year2year" || frmoption=="age2year") {
			age1 <- spell[1,begin]
		}
		# if we need to convert years to ages, we need the birthyear
		if (frmoption=="year2age") {
			if (length(birthyrid1)==1 && all(pdata=="auto")) {
				birthy <- spell[1,begin]
				if (!is.na(birthy)) age1 <- 0 else age1 <- NA ## gr Aug 21 fixed for NA birthy
			}
			else if (all(lid %in% birthyrid1)) {
    			birthy <- birthyr1[birthyrid1==lid[i]]
    			#if(nbseq<40){
                    #print(paste("spell 1 = ", spell[1,begin]))
    			    #print(paste("birthyr = ", birthy))
                #}
    			age1 <- spell[1,begin] - birthy
    			if (!is.na(age1) & age1 < 0) { ## first spell starts before birthyr
                    nstartbefore <- nstartbefore + 1
                    istartbefore <- c(istartbefore, i)
                    if (spell[1,end] >= birthy) {
                      #if (nstartbefore < 10)
                      #  msg.warn("First spell of sequence",i, "starts before birth year! Start set as birth year")
                      spell[1,begin] <- birthy
                      age1 <- 0
                    }
                    #else if (nstartbefore < 10)
                    #  msg.warn("First spell of sequence",i, "occurs before birth year! Start set as NA")
                    #if (nstartbefore == 10)
                    #  msg.warn("...")
                    else {
                      nendbefore <- nendbefore + 1
                      iendbefore <- c(iendbefore, i)
                    }
                }
			}
			else {
				stop(" [>] pdata must either contain a birth year per individual or be set as \"auto\"")
			}

		}
		# if we convert from ages to years, we need the birthyear, but do not need to substract it to the time of beginning
		if (frmoption=="age2year") {
			birthy <- birthyr1[birthyrid1==lid[i]]
			#print("birthyr")
			#print(birthy)
			age1 <- spell[1,begin]
		}
		if (is.na(age1)) {
      nemptseq <- nemptseq + 1
      iemptseq <- c(iemptseq, i)
      #if (nemptseq < 11) {
			#   msg.warn("start time is missing for case ", i,", empty sequence created")
      #} else if (nemptseq == 11)
			#   msg.warn("...")
			age1 <- -1
		}

		# we fill the line with NAs
		seqresult[i,1:(limit)] <- c(rep(NA,(limit)))

	  if (age1 >= 0) {
			if (idxmax>0) {
				# by default, the most recent episode erases the one before
				if(overwrite==TRUE) {
					spelllist <- 1:idxmax
				}
				# if we want the opposite, we just go from the last to the first episode
				else {
					spelllist <- idxmax:1
				}
				# for each spell
				for (j in spelllist) {
					#####################
					# definition of starting and ending point of the spell

					# spell are allready in age format, and we want age sequences
					if(frmoption=="age2age") {
						sstart <- spell[j,begin]
						sstop <- spell[j,end]
					}
					if(frmoption=="age2year") {
						sstart <- (birthy-tmin) +  (spell[j,begin]) +1
						sstop <- (birthy-tmin) +  (spell[j,end]) +1
					}
					# spell are in year format, and we want year sequences
					if(frmoption=="year2year") {
						sstart <- (spell[j,begin] - tmin)+1
						sstop <- (spell[j,end] - tmin)+1
						#print(sstart)
						#print(sstop)
					}

					# spell are in year format, and we want age sequences
					if(frmoption=="year2age") {
						sstart <- spell[j,begin] - birthy + 1
						sstop <- spell[j,end] - birthy + 1
					}
					if(is.na(sstart) | is.na(sstop)) {
						msg.warn("skipping episode ",j, " for case ", i, " due to missing start/end time")
					}
					else {
					#######################
					# fillblanks option
					# if fillblanks is not null, gaps between episodes are filled with its value
						if (!is.null(fillblanks)) {
						#print(fillblanks)
							if (j>1) {
							# for every episode after the first one, we check if there is a gap
              # between the previous and current episodes.
								if(frmoption=="age2age" || frmoption=="age2year") {
									previousend <- spell[j-1,end]
								}
								if(frmoption=="year2age") {
									previousend <- spell[j-1,end] - birthy
								}
								if(frmoption=="year2year") {
									previousend <- (spell[j-1, end] - tmin)+1
								}
								if (sstart != previousend && sstart != (previousend+1)) {
									dur <- sstart - (previousend+1)
									if (dur>0 & (sstart-1 < limit) && sstart > 0 && spell[j-1,end] > 0) {
										seqresult[i,(previousend+1):(sstart-1)] <- rep(fillblanks, dur)
									}
								}

							}
						}
					#
					#########################

					#########################
					# conversion from episode to subsequence
						dur <- sstop - sstart + 1
					# we check if all values look normal
						if (dur >= 0 && sstop > 0 && sstart >= 0) {
							state <- spell[j,status]
							if (!is.na(state)) {
							# if dur == 0, it means the individual stays in the state only one year
							# if (dur == 0 && (sstop < limit) ) {
							#	seqresult[i,sstart] <- state
              # }

									if(sstop <= limit) {
									# if the sequence begins at age 0, we delete the first state
									# if (sstart==0) {
									#	sstart <- sstart+1
									#	dur <- dur -1
									#	}
										seqresult[i,sstart:sstop] <- rep(state, dur)
									}
                  else {
                    if (i != ii) {
                      ntrunc <- ntrunc + 1
                      itrunc <- c(itrunc, i)
                      ii <- i
                    }
                    #if (maxlength.warn){
                    #  msg.warn("Some sequences exceed limit length and are truncated")
                    #  msg.warn("Sequence",i,"truncated at limit",limit)
                    #  maxlength.warn <- FALSE
                    #}
                    if (sstart <= limit) {
                      tdur <- limit - sstart + 1
                      seqresult[i,sstart:limit] <- rep(state, tdur)
                    }
                  }
					   	}
						}
					}
				}
			 }
		}
	}
  if (nstartbefore > 0)
    msg.warn(nstartbefore,"cases with a spell starting before birth year. Indexes in attribute 'issues'")
  if (nendbefore > 0)
    msg.warn(nendbefore,"empty sequences because of spell occurring before birth year. Indexes in attribute 'issues'")
  if (nemptseq > 0)
    msg.warn(nemptseq,"empty sequences because of missing start time. Indexes in attribute 'issues'")
  if (ntrunc > 0)
    msg.warn(ntrunc,"sequences truncated at limit",limit,". Indexes in attribute 'issues'")
  seqresult <- as.data.frame(seqresult)
  if(is.factor(status.orig)) {
    for (k in 1:(limit)) {
      if(is.null(fillblanks)) {
	      seqresult[,k] <- factor(seqresult[,k], levels=1:nlevels(status.orig), labels=levels(status.orig))
			}
			else {
				seqresult[,k] <- factor(seqresult[,k], levels=1:(nlevels(status.orig)+1), labels=c(levels(status.orig), fillblanksF))
			}
		}
  }
  names(seqresult) <- names.seqresult
  attr(seqresult,"issues") <- list(truncated = itrunc, empty.seq = iemptseq, start.before.birth = istartbefore,
            spell.before.birth = iendbefore)

	## setting id as rowname
	row.names(seqresult) <- lid

	return(seqresult)
}
