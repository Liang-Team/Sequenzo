

seqnull <- function(seqdata, model=c("combined", "duration", "sequencing", "stateindep", "Markov", "userpos"),
					imp.trans=NULL, imp.trans.limit=-1, trate="trate", begin="freq", time.varying=TRUE, weighted=TRUE){
	model <- model[1]
	
	if(any(seqdata==attr(seqdata, "nr"))){
		stop(" [!] Missing data not (yet) supported in seqnull(). For now, missing value can be recoded as a specific state before running the analysis.\n")
	}
	if(model %in% c("combined", "duration", "sequencing")){
		return(seqnullspell(seqdata, randomize=model, impossibleTrans=imp.trans, impossibleTransTreshold=imp.trans.limit))
	}
	if(model=="stateindep"){
		return(seqnullpos(seqdata, prob="freq", time.varying=TRUE))
	}
	if(model== "Markov"){
		return(seqnullpos(seqdata, prob="trate", time.varying=FALSE))
	}
	if(model== "userpos"){
		return(seqnullpos(seqdata, prob=trate, begin=begin, time.varying=time.varying, weighted=weighted))
	}else{
		stop(" [!] Unknow seqnull model.\n")
	}

}
seqnullspell <- function(seqdata,  randomize, impossibleTrans, impossibleTransTreshold)
#method.first="freq", method.dss="obs", method.dur="state")
{
	if(randomize=="combined"){
		seq <- "random"
		duration <- "sample"
	} else if(randomize=="duration"){
		seq <- "keep"
		duration <- "random"
	}
	else if(randomize=="sequencing"){
		seq <- "sequencing"
		duration <- "sequencing"
	}
	
	dss <- seqdss(seqdata)
	dur <- seqdur(seqdata)
	vfunc <- function(x) as.vector(as.matrix(x))
	spells <- droplevels(subset(data.frame(state=vfunc(dss), dur=vfunc(dur)), !is.na(dur)))

	if(randomize=="sequencing"){
		ss <- sample(nrow(spells), size=nrow(seqdata)*ncol(seqdata), replace=TRUE)
		dur2 <- matrix(spells$dur[ss], nrow=nrow(seqdata), ncol=ncol(seqdata))
		dss2 <- matrix(as.character(spells$state[ss]), nrow=nrow(seqdata), ncol=ncol(seqdata))
	}else{
	
		if(seq %in% c("random")){
			stateprop <- prop.table(table(spells$state))[alphabet(dss)]
			
			tr <- matrix(stateprop, ncol=length(stateprop), nrow=length(stateprop), byrow=TRUE)
			dimnames(tr) <- list(alphabet(dss), alphabet(dss))
			diag(tr) <- 0
			if(impossibleTransTreshold >= 0){
				tr[seqtrate(dss) <= impossibleTransTreshold] <- 0
			}
			if(!is.null(impossibleTrans)){
				for(tt in names(impossibleTrans)){
						tr[tt, impossibleTrans[[tt]]] <- 0
				}
			}
			tr <- tr/rowSums(tr)
			dss2 <- matrix(NA_character_, nrow=nrow(seqdata), ncol=ncol(seqdata))
			dss2[, 1] <- sample(names(stateprop), size=nrow(dss2), replace=TRUE, prob=stateprop)
			for(i in 2:ncol(dss2)){
				for(a in names(stateprop)){
					cond <- dss2[, i-1]==a
					ncond <- sum(cond)
					if(ncond>0){
						dss2[cond, i] <- sample(names(stateprop), size=ncond, replace=TRUE, prob=tr[a, ])
					}
				}
			}
		}else{
			dss2 <- as.matrix(dss)
			dss2[dss2==attr(seqdata, "void")] <- NA_character_
			
		}
		if(duration=="sample"){
			dur2 <- matrix(sample(spells$dur, size=nrow(seqdata)*ncol(seqdata), replace=TRUE), nrow=nrow(seqdata), ncol=ncol(seqdata))
			
		}else{
			nspell <- rowSums(!is.na(dss2))
			durprop <- matrix(runif(ncol(dss2)*nrow(dss2)), nrow=nrow(dss2), ncol=ncol(dss2))
			durprop[is.na(dss2)] <- NA
			durprop <- 1+(ncol(seqdata) -nspell)*durprop/rowSums(durprop, na.rm=TRUE)	
			dur2 <- floor(durprop)
			durrest <- durprop-dur2
			durtot <- rowSums(dur2, na.rm=T)
			cond <- (t(apply(durrest, 1, order, decreasing=TRUE))<=(ncol(seqdata)-durtot))
			dur2[cond] <- dur2[cond]+1
		}	
	}
	
	
	dssToSts <- function(i){
		cond <- !is.na(dur2[i, ])
		return(rep(dss2[i, cond], times=as.integer(dur2[i, cond]))[1:ncol(seqdata)])
	}
	seqdata2 <- t(sapply(1:nrow(seqdata), dssToSts))
	for(i in 1:ncol(seqdata2)){
		seqdata[, i] <- factor(seqdata2[, i], levels=levels(seqdata[, i]))
	}
	##seqdata[] <- seqdata2
	return(seqdata)
}



seqnullpos <- function(seqdata, prob="trate", time.varying=TRUE, begin="freq", weighted=TRUE){
	seqasnum <- function (seqdata, with.missing = FALSE) 
	{
		mnum <- matrix(NA, nrow = seqdim(seqdata)[1], ncol = seqdim(seqdata)[2])
		rownames(mnum) <- rownames(seqdata)
		colnames(mnum) <- colnames(seqdata)
		statl <- attr(seqdata, "alphabet")
		if (with.missing) 
			statl <- c(statl, attr(seqdata, "nr"))
		for (i in 1:length(statl)) {
			mnum[seqdata == statl[i]] <- i - 1
		}
		return(mnum)
	}

	sl <- seqlength(seqdata)
	maxage <- max(sl)
	nbtrans <- maxage -1
	agedtr <- vector(mode="list", length=maxage)
	
	## On ajoute 1 pour que les codes correspondent aux index R (commence a 1)
	seqdatanum <- seqasnum(seqdata)+1
	nbstates <- max(seqdatanum)
	## User defined begin frequencies
	if(is.numeric(begin)){
		if (length(begin)!=nbstates) {
			stop("Begin frequencies should be a numeric vector of length ", nbstates)
		}
		message(" [>] Using user defined frequencies as starting point")
		firstfreq <- begin
	}
	##Compute from data
	else if (is.character(begin) && begin=="freq") {
		message(" [>] Using data defined frequencies as starting point")
		firstfreq <- seqstatd(seqdata, weighted=weighted)$Frequencies[, 1]
	}
	else if (is.character(begin) && begin=="ofreq") {
		message(" [>] Using overall data defined frequencies as starting point")
		firstfreq <- seqstatf(seqdata, weighted=weighted)$Percent/100
	}
	else {
		stop("Unknow method to compute starting frequencies")
	}
	
	###Automatic method to compute transition rates
	if (is.character(prob)) {
		if (prob=="trate") {
			if (time.varying) {
				message(" [>] Using time varying transition rates as probability model")
				agedtr <- seqtrate(seqdata, time.varying=TRUE, weighted=weighted)
			}
			else {
				message(" [>] Using global transition rates as probability model")
				agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
				tr <- seqtrate(seqdata, weighted=weighted)
				for (i in 1:nbtrans) {
					agedtr[,,i] <- tr
				}
			}
		}
		else if (prob=="freq") {
			## On cree quand meme une matrice de transition (qui ne depend pas de l'etat precedant)
			## On peut ainsi utiliser le meme algorithme
			message(" [>] Using time varying frequencies as probability model")
			agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
			if (time.varying) {
				freqs <- seqstatd(seqdata, weighted=weighted)$Frequencies
				for (i in 1:nbtrans) {
					for (j in 1:length(freqs[, i+1])) {
							agedtr[, j,i] <- freqs[j, i+1]
					}
				}
			}
			else {
				message(" [>] Using global frequencies as probability model")
				freqs <- seqstatf(seqdata, weighted=weighted)$Percent/100
				for (i in 1:nbtrans) {
					for (j in 1:length(freqs)) {
						agedtr[, j,i] <- freqs[j]
					}
				}
			}
		}
		else {
			stop("Unknow method to compute transition rate")
		}
	}
	## User defined transition rates
	else{
		if(is.array(prob)){
			if(length(dim(prob)) == 3) {
				##Correct dimensions
				if(any(dim(prob)!=c(nbstates, nbstates, nbtrans))){
					stop("Transition rate should be an array of size (state x state x transition) ",
						nbstates,"x", nbstates, "x", nbtrans)
				}
				message(" [>] Using user defined time varying transition rates as probability model")
				agedtr <- prob
			} else if (length(dim(prob)) == 2) {
				message(" [>] Using user defined global transition rates as probability model")
				if(any(dim(prob)!=c(nbstates, nbstates))){
					stop("Transition rate should be a matrix of size (state x state) ",
						nbstates,"x", nbstates)
				}
				agedtr <- array(0, dim=c(nbstates, nbstates, nbtrans))
				for (i in 1:nbtrans) {
					agedtr[,,i] <- prob
				}
			}
			else {
				stop("Transition rate should be an array of size (state x state x transition) ",
						nbstates,"x", nbstates, "x", nbtrans, " or a matrix of size (state x state) ",
						nbstates,"x", nbstates)
			}
		}
		else {
			stop("Unknow method to compute transition rate")
		}
	}
	randomstate <- function(n, probs){
		return(sample.int(nbstates, n, replace=TRUE, prob=probs))
		pp <- numeric(length(probs)+1)
		pp[1] <- 0
		
		for(i in 1:length(probs)){
			pp[i+1] <- pp[i]+probs[i]
		}
		removed <- which(probs==0)
		if(length(removed)>0){
			return(cut(runif(n), breaks=pp[-(removed+1)], include.lowest=T, labels=(1:length(probs)))[-removed])
		}
		return(cut(runif(n), breaks=pp, include.lowest=T, labels=1:length(probs)))
	}
	## Now generate the sequences
	nullseq <- matrix(-1, nrow=nrow(seqdata), ncol=ncol(seqdatanum))
	nullseq[, 1] <- randomstate(nrow(seqdata), firstfreq)
	
	for(i in 2:ncol(seqdatanum)){
		for(ss in 1:nbstates){
			cond <- nullseq[, i-1]==ss
			ncase <- sum(cond)
			if(ncase>0){
				nullseq[cond, i] <- randomstate(ncase, agedtr[ss, , i-1])
			}
		}
	}
	ret <- seqdef(nullseq, alphabet=1:nbstates, labels=attr(seqdata, "labels"), states=alphabet(seqdata))
	return(ret)
}
