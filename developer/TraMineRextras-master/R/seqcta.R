seqcta <- function(seqdata, subseq=5, time=NULL, event=NULL, initial.state=NULL, covar=NULL){
	sl <- seqlength(seqdata)
	timevent <- time
	if(!is.null(timevent)){
		if(!is.numeric(timevent)||nrow(seqdata)!=length(timevent)){
			stop(" [!] time should be a numeric vector of the same length as seqdata")
		}
		if(!is.null(event)){
			if(!is.logical(event)||nrow(seqdata)!=length(event)){
				stop(" [!] event.occurrence should be a logical vector of the same length as seqdata")
			}
		}
		else{
			event <- !is.na(timevent)
		}
	}else{ # We compute it from data and initial state.
		if(is.null(initial.state)||!(initial.state %in% alphabet(seqdata))){
			stop(" [!] If time and event are not provided, initial.state should be specified and be one state of the alphabet of seqdata.")
		}
		timevent <- seqdur(seqdata)[, 1]
		timevent[seqdata[, 1]!=initial.state] <- 0
		event <- timevent < sl
	}
	## Now we adjust censoring time
	
	maxtime <- sl - subseq
	censored <- timevent>maxtime
	timevent[censored] <- maxtime[censored]
	event[censored] <- FALSE
	##Now we computed the 
	
	suppressMessages(srs <- seqformat(seqdata, from="STS", to="SRS"))
	srs <- srs[, c(1:2, (ncol(srs)-subseq+1):ncol(srs))]
	srs$event <- event[srs$id]
	timelimit <- timevent[srs$id]+subseq
	srs$event <- srs$event & srs$idx==timelimit
	srs <- subset(srs, srs$idx<=timelimit)
	colnames(srs) <- c("id", "time", paste0("T", 1:subseq), "event")
	srs[!srs$event, c(-1, -2, -ncol(srs))] <- NA
	srs$time <- srs$time - subseq+1
	srs <- subset(srs, time>0)
	srs$lastobs <- srs$time == ave(srs$time, srs$id, FUN=max)
	srs <- data.frame(srs[, c("id", "time", "event", "lastobs", paste0("T", 1:subseq))])
	if(!is.null(covar)){
		if(!is.data.frame(covar)||nrow(covar)!=nrow(seqdata)){
			stop(" [!] covar should be a data frame with the same number of rows as seqdata")
		}
		srs <- cbind(srs, covar[srs$id,])
	}
	return(srs)
}
