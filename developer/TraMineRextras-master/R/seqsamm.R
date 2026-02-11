
## The extractsubseq function generates a person-period dataset based on the sequence data. 
## At each time position, the subsequence for the next time units is also provided.  
## It accepts the following arguments:
## seqdata: sequence data as produced by seqdef (see TraMineR documentation)
## sublength: the length of the subsequence to consider.
## id: an optional vector specifying an id for each sequence. If not specified, an id is generated based on row numbers.
## covar: an optional data frame providing time-invariant covariable information for each cases. Rownames should match the id (if specified)
##
## The function returns a "subseq" object. This is a person-period data frame storing the following variables:
##  - id: id or row number.
##  - time: time elapsed since the beginning of the sequence
##  - spellbegin: beginning of the current spell
##  - spelltime: time elapsed since the beginning of the spell
##  - transition: dummy indicating whether there is a transition.
##  - subsequence (named s.1 ... s.x): the subsequences (the number of variable depend on subsequence length),
##  - If provided, covariables are located at the end of the data.frame.
## A subseq object also stores some additional informations, such as the typology (see function setTypology)

seqsamm <- function(seqdata, sublength, covar=NULL){
	minlength <- sublength
	sname <- paste("s", 1:sublength, sep=".")
	maxmiss <- sublength-minlength
	spellbegin <- rep(1, nrow(seqdata))
	
	allsubseq <- list()
	id <- 1:nrow(seqdata)
	for(tt in 1:(ncol(seqdata)-sublength)){
		subseq <- seqdata[, tt:(tt+sublength-1)]
		cond <- rowSums(subseq==attr(seqdata, "void"))<=maxmiss
		colnames(subseq) <- sname
		transition <- (subseq[, 1]!=subseq[, 2])
		if(tt>1){
			condSpellReset <- seqdata[, tt-1]!=seqdata[, tt]
			spellbegin[condSpellReset] <- tt
		}
		subseq <- data.frame(id=id, time=tt,begin=spellbegin, spell.time=tt-spellbegin, transition=transition, subseq)
		allsubseq[[tt]] <- subseq[cond, ]
	}
	ret <- do.call(rbind, allsubseq)
	if(!is.null(covar)){
		ret <- cbind(ret, covar[ret$id, ])
	}
	ret <- ret[order(ret$id, ret$time), ]
	attr(ret, "stslist") <- ret[, sname]
	attributes(attr(ret, "stslist"))[c("start", "missing", "void", "nr", "alphabet", "labels", "cpal", "missing.color", "xtstep", "tick.last", "Version", "class")] <- attributes(seqdata)[c("start", "missing", "void", "nr", "alphabet", "labels", "cpal", "missing.color", "xtstep", "tick.last", "Version", "class")]
	attr(ret, "spell") <- alphabet(seqdata)
	attr(ret, "typology") <- rep("None", nrow(ret)) 
	attr(ret, "sname") <- sname
	attr(ret, "sublength") <- sublength
	class(ret) <- c("SAMM", class(ret))
	return(ret)
}

plot.SAMM <- function(x, type="d", ...){
	seqdata <- attr(x, "stslist")[x$transition,]
	group <- x[x$transition, attr(x, "sname")[1]]
	levels(group) <- paste("Transition out of", levels(group))
	seqplot(seqdata, group=group, type=type, ...)
}

## getChangeSubseq return a sequence object containing the subsequences that follows a given spell
## It accepts the following arguments:
##  subseq: A subseq object (see extractSubseq)
##  spell: the ending spell 
seqsammseq <- function(samm, spell){
	cond <- samm[, attr(samm, "sname")[1]]==spell& samm$transition
	subseqdata <- attr(samm, "stslist")[cond,]
	return(subseqdata)
}

## setTypology can be used to set the typology of subsequence following a given spell
##  - subseq: A subseq object (see extractSubseq)
##  - spell: the ending spell
##  - typology: The typology that should be used in subsequent analysis
## The function returns the updated object.
"typology<-" <- function(samm, spell, value){
	return(samm)
}



## getEHAData returns a data frame suitable to estimate the hazard of following a given type after a given spell
##  - subseq: A subseq object (see extractSubseq)
##  - type: the type of subsequence to consider (the spell is automatical deducted from the type)
## The function returns a data.frame with an additional "event" column suitable to estimate a multistate model.

seqsammeha <- function(samm, spell, typology, persper=TRUE){
	cond <- with(samm, s.1==spell&transition)
	attr(samm, "typology")[cond] <- as.character(typology)
	spellcond <- with(samm, s.1==spell)
	ppdata <-subset(samm, spellcond)
	ppdata$SAMMtypology <- factor(attr(samm, "typology")[spellcond])
	ppdata$SAMMtypology <- relevel(ppdata$SAMMtypology , "None")
	ppdata$lastobs <-  ave(ppdata$spell.time, paste(ppdata$id, ppdata$begin), FUN = max)==ppdata$spell.time

	if(is.factor(typology)){
		types <- levels(typology)
	}else{
		types <- unique(typology)
	}
	ret <- matrix(0, nrow=nrow(ppdata), ncol=length(types))
	colnames(ret) <- types
	for(tt in types){
		ret[, tt] <- ppdata$SAMMtypology==tt
	}
	colnames(ret) <- paste0("SAMM", colnames(ret))
	ppdata <- cbind(ppdata, ret)

	if(!persper){
		return(subset(ppdata, ppdata$lastobs))
	}
	return(ppdata)
}

