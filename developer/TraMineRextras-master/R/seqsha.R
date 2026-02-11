seqsha <- function(seqdata, time, event, include.present=FALSE, align.end=FALSE, covar=NULL){

	## Person-Level Person-Period Converter Function
	PLPP <- function(data, id="id", period="time", event="event") {
	  ## Data Checking and Verification Steps
	  if (any(is.na(data[, c(id, period, event)]))) {
		stop(" [!] cannot handle missing data in the time or event variables")
	  }

	  ## Do the conversion
	  
		  index <- rep(1:nrow(data), data[, period])
		  idmax <- cumsum(data[, period])
		  dat <- data[index, ]
		  dat[, period] <- ave(dat[, period], dat[, id], FUN = seq_along)
		  dat[, event] <- FALSE
		  dat[idmax, event] <- data[, event]
	  
	  rownames(dat) <- NULL
	  return(dat)
	}
	
	basetime <- data.frame(id=1:nrow(seqdata), time=time, event=event)
	
	persper <- PLPP(basetime, "id", "time", "event")
	
	sdata <- as.matrix(seqdata)
	sdata[is.na(sdata)] <- "NA_orig"
	age <- persper$time
	ma <- max(age)
	if(ma>ncol(seqdata)){
		stop(" [!] Maximum time of event occurrence is higher than the longuest sequence!")
	}
	past <- matrix(as.character(NA), ncol=ncol(seqdata), nrow=nrow(persper))

	if(align.end){
		start <- ifelse(include.present, 1, 2)
		for(aa in start:ma){
			cond <- age == aa
			idsA <- persper$id[cond]
			if(include.present){
				past[cond, (ncol(seqdata)-aa+1):ncol(seqdata)] <- sdata[idsA, 1:(aa)]
			}
			else{
				past[cond, (ncol(seqdata)-aa+2):ncol(seqdata)] <- sdata[idsA, 1:(aa-1)]
			}
			
			# for(aa2 in 1:aa){
				# idsA <- persper$id[cond]
				# past[cond, ncol(seqdata)+1-aa2] <- sdata[idsA, aa2]
			# }
		}
		colnames(past) <- paste0("Tm", ncol(past):1)
	}else{
		for(aa in 1:ma){
			if(include.present){
				cond <- age > aa
			}else{
				cond <- age >= aa
			}
			idsA <- persper$id[cond]
			past[cond, aa] <- sdata[idsA, aa]
		}
		colnames(past) <- colnames(seqdata)[1:ma]
	}
	alldata <- cbind(persper, past)
	if(!is.null(covar)){
		alldata <- cbind(alldata, covar[alldata$id,])
	}
	return(alldata)

}
