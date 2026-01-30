#######################################################
###Compare groups and returns most discriminating func
#######################################################

#seqecmpgroup.survival <- function(time, event, var, stest) {
#	library(survival)
#	test <- survdiff(Surv(time, event) ~ var, rho=stest)
#	pval <- 1-pchisq(test$chisq,length(test$n)-1)

#	pval
#}

#seqecmpgroup.chisq<-function(group,seqp){
#  chi<- chisq.test(group,seqp)
#  return(chi$statistic)
#}

seqecmpgroup <- function(subseq, group, method="chisq", pvalue.limit=NULL, weighted=TRUE){
	## If non weighted, we just change the weights to 1
	if(!weighted) {
		www <- seqeweight(subseq$eseq)
		totseq <- length(subseq$eseq)
		ww <- as.double(rep(1, totseq))
		seqeweight(subseq$eseq) <- ww
	} else {
		ww <- seqeweight(subseq$eseq)
		totseq <- sum(ww)
	}
	seqecmpgroup.chisq <- function(index, group, seqmatrix, bonferroni, ntest){
		sp<-sum(seqmatrix[ , index])
		## If we have no occurence of a subsequence chisq.test will fail to compute
		if(sp>0 && sp<length(group)){
			suppressWarnings(chi <- chisq.test(xtabs(ww~group+seqmatrix[ , index])))
			if (bonferroni) {
				chi$p.value <- (1-(1-chi$p.value)^ntest)
			}
			resid <- as.list(chi$residuals[, "1"])
			names(resid) <- paste("Resid", names(chi$residuals[, "1"]), sep=".")
			freq <- as.list(chi$observed[, "1"]/rowSums(chi$observed))
			names(freq) <- paste("Freq", names(chi$residuals[, "1"]), sep=".")
			return(data.frame(p.value=chi$p.value, statistic=chi$statistic,
				index=index, freq, resid, check.names=FALSE))
			#return(data.frame(p.value=chi$p.value,statistic=chi$statistic,index=index))
		} else {
			freq <- numeric(length(levels(group)))
			names(freq) <- paste("Freq", levels(group), sep=".")
			resid <- numeric(length(levels(group)))
			names(resid) <- paste("Resid", levels(group), sep=".")

			freq[] <- sp/length(group)
			return(data.frame(p.value=1.0, statistic=0, index=index,
				as.list(freq), as.list(resid), check.names=FALSE))
		}
#    return(data.frame(p.value=NA,statistic=NA,index=index))
	}
	if(is.null(pvalue.limit)) {
		pvalue.limit<- 2
	}
	if(!is.subseqelist(subseq)) {
		stop(" [!] subseq should be a subseqelist")
	}
	group <- factor(group)
	if (method=="bonferroni") {
		bonferroni <- TRUE
		method <- "chisq"
	}
	else {
		bonferroni <- FALSE
	}
	if (method=="chisq") {
		testfunc <- seqecmpgroup.chisq
		ntest <- length(subseq$subseq)
		seqmatrix <- seqeapplysub(subseq, method="presence")
		testfunc.arg <- list(group=group, bonferroni=bonferroni,
			ntest=ntest, seqmatrix=seqmatrix)
		decreasing<-TRUE
	}
	else {
		stop(" [!] This method is not (yet) implemented")
	}
	res <- data.frame()
	for (i in 1:length(subseq$subseq)) {
		testfunc.arg$index <- i
		stat <- do.call(testfunc, testfunc.arg)
		res <- rbind(res,stat)
	}
	## Keeped number of subsequence
	subseqnum <- 1:sum(res[,1]<=pvalue.limit)
	## Finding the index and order of the subsequence
	cres <- order(as.double(res[,2]), decreasing = decreasing)[subseqnum]#[!is.na(as.double(res$stat))]
	data <- data.frame(Support=as.data.frame(subseq$data[cres,"Support"], optional=TRUE), res[cres,], check.names=FALSE)
	rownames(data) <- 1:nrow(data)
	ret <- createsubseqelist(subseq$eseq, subseq$constraint, subseq$subseq[cres], data=data, type=method)
	ret$labels <- levels(group)
	ret$bonferroni <- list(used=bonferroni, ntest=ntest)
	class(ret) <- c("subseqelistchisq",class(ret))
	if(!weighted) {
		seqeweight(subseq$eseq) <- www
	}
	return(ret)
}



plot.subseqelistchisq <- function(x, ylim = "uniform", rows = NA, cols = NA,
  resid.levels = c(0.05,0.01), cpal = brewer.pal(1 + 2 * length(resid.levels), "RdBu"),
  vlegend = NULL, cex.legend = 1, ptype = "freq", legend.title = NULL,
  with.legend=TRUE, residlevels, legendcol, legend.cex, ...) {

  TraMineR.check.depr.args(alist(resid.levels = residlevels, vlegend = legendcol, cex.legend = legend.cex))

	if(!inherits(x,"subseqelistchisq")) {
		stop(" [!] x should be a result of seqecmpgroup")
	}
	nplot<-length(x$labels)
	#  print(ylim)
	pvalue.levels <- resid.levels
	if (x$bonferroni$used) {
		resid.levels <- (1- (1-resid.levels)^(1/x$bonferroni$ntest))
	}
	resid.levels <- sort(-qnorm(resid.levels))

	#print(cpal)
	residbreaks <- c(-Inf, -sort(resid.levels), sort(resid.levels), Inf)
	lout <- TraMineR.setlayout(nplot, rows, cols, with.legend=with.legend, "all")
	## Save all current settings
	savepar <- par(no.readonly = TRUE)
	on.exit(par(savepar))
	layout(lout$laymat, heights=lout$heights, widths=lout$widths)

	if(ptype=="resid"){
		baseIndex <- 4 + nplot
	} else {
		baseIndex <- 4
	}
	if(ylim=="uniform"){
		ylim <- c(min(min(x$data[ , (baseIndex+1):(baseIndex+nplot)]), 0),max(x$data[ , (baseIndex+1):(baseIndex+nplot)]))
	}
	for(i in 1:nplot){
		ccol <- as.character(cut(x$data[ , 4+nplot+i], breaks=residbreaks, labels=cpal))
		plot.subseqelist(x, freq=x$data[,baseIndex+i], col=ccol, main=x$labels[i], ylim=ylim, ...)
	}
    if (with.legend) {
	    par(mar = c(1, 1, 0.7, 1) + 0.1, xpd=FALSE)
        if (is.null(legend.title)){
           legend.title <- "Color by sign and significance of Pearson's residual"
           }
    	#on.exit(par(savepar))
    	plot(0, type = "n", axes = FALSE, xlab = "", ylab = "")
    	title(main=legend.title, cex=cex.legend)
    	legncol <- length(c(paste("-", rev(resid.levels)), "neutral", resid.levels))
    	if(is.null(vlegend) && lout$legpos=="center"){
    		legncol <- 1
    	}
    	else if(!is.null(vlegend) && vlegend){
    		legncol <- 1
    	}
    	legend(lout$legpos,
    			# inset=c(0,leg.inset),
    			legend=c(paste("Negative", rev(pvalue.levels)), "neutral", paste("Positive", pvalue.levels)),
    			fill=cpal,
    			ncol=legncol,
    			cex=cex.legend,
    			bty="o"
    		)
    }
}
