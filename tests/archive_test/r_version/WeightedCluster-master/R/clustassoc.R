clustassoc <- function(clustrange, diss, covar,  weights = NULL){
	if(any(dim(diss)!=length(covar))){
		stop(" [!] diss should be a dissimilarity matrix or a dist object and covar should be a variable with one value per observation.")
	}
	nullMM <- dissmfacw(diss~covar, R=1, data=NULL, weights=weights)
	if(is.numeric(covar)){
		nullMod <- lm(covar~1, weights=weights)
	}else{
		nullMod <- nnet::multinom(covar~1, weights=weights, trace=FALSE)
	}
	nullBIC <- BIC(nullMod)
	
	stat <- clustrange$stats[, 1:3]
	colnames(stat) <- c("Unaccounted", "Remaining", "BIC")
	
	for(i in 1:ncol(clustrange$clustering)){
		clustf <- factor(clustrange$clustering[, i])
		mm <- dissmfacw(diss~clustf+covar, R=1, data=NULL, weights=weights)
		stat$Remaining[i] <- mm$mfac[2,3]
		stat$Unaccounted[i] <- mm$mfac[2,3]/(mm$mfac[3,3]-mm$mfac[1,3])
		#stat$assoc[i] <- assocstats(table(clustf, covar))$cramer
		if(is.numeric(covar)){
			mod <- lm(covar~clustf, weights=weights)
		}else{
			mod <- multinom(covar~clustf, weights=weights, trace=FALSE)
		}
		stat$BIC[i] <- BIC(mod)
	}
	stat <- rbind(c(1, nullMM$mfac[1,3], nullBIC ), stat)
	rownames(stat)[1] <- "No Clustering"
	#stat$BICgain <- stat$BIC-stat$BIC[1]
	stat$numcluster <- c(1, clustrange$kvals)
	class(stat) <- c("clustassoc", class(stat))
	return(stat)
}

plot.clustassoc <- function(x, stat=c("Unaccounted", "Remaining", "BIC"), type="b", ...){
	titles <- c("residualProp"="Unreproduced proportion of the association", "residual"="Unexplained association", "BIC"="BIC", "BICgain"="Gain in BIC")
	plot(x$numcluster, x[, stat[1]], ylab=titles[stat[1]], xlab="Number of clusters", type=type, ...)
}


