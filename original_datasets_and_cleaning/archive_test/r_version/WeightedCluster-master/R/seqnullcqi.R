normstatcqi <- function(bcq, stat, norm=TRUE){
	
	origstat <- bcq$clustrange$stats[, stat]
	nullstat <- bcq$stats[[stat]]
	
	#normstat <- rbind(nullstat, origstat)
	if(norm){
		for(i in seq_along(origstat)){

			mx <- mean(nullstat[, i])
			sdx <- sd(nullstat[, i])
			nullstat[ , i] <- (nullstat[, i]-mx)/sdx
			origstat[i] <- (origstat[i]-mx)/sdx
		}
	}
	alldatamax <- apply(nullstat, 1, max)#as.vector(xx)
	sumcqi <- list(origstat=origstat, nullstat=nullstat, alldatamax=alldatamax)
	return(sumcqi)

	

}
confcqi <- function(nullstat, quant, n){
	alpha <- (1-quant)/2
	#calpha <- alpha+(alpha-1)/n
	#print(c(calpha, alpha))
	#minmax <- quantile(nullstat, c(calpha, 1-calpha))
	minmax <- quantile(nullstat, c(alpha, 1-alpha))
	return(minmax)
}
plotncqdensity2 <- function(bcq, stat, quant=NULL, norm=FALSE, maxt=TRUE,  ...){
	cq <- bcq$clustrange
	
	sumcqi <- normstatcqi(bcq, stat=stat, norm=norm)
	#alldata <- as.vector(sumcqi$nullstat)
	allrange <- range(c(sumcqi$origstat, sumcqi$alldatamax))
	internalplot <- function(origstat, alldata, col, add=FALSE, seg=TRUE){
	
		kvals <- cq$kvals
		
		dens <- density(alldata)
		#plot(dens, ...)
		if(add){
			lines(dens, col=adjustcolor(col, alpha.f=.9),...)
		}else{
			plot(dens, xlim=allrange, col=adjustcolor(col, alpha.f=.9),...)
		}
		if(!is.null(quant)){
			minmax <- confcqi(alldata, quant, bcq$R)
			x1 <- min(which(dens$x >= minmax[1]))  
			x2 <- max(which(dens$x <  minmax[2]))
			with(dens, polygon(x=c(x[c(x1, x1:x2, x2)]), y= c(0, y[x1:x2], 0), col=adjustcolor(col, alpha.f=.15), border=adjustcolor(col, alpha.f=.9)))
			print(minmax)
		}
		if(seg){
			adj <- (seq_along(kvals)-1)/(length(kvals)-1)*0.6+0.2
			segments(origstat, 0, origstat, max(dens$y)*adj, col="black")
			text(origstat, max(dens$y)*(adj+0.05), labels=kvals, col="black")
		}
	}
	
		internalplot(sumcqi$origstat, sumcqi$alldatamax, col="#E41A1C")
		#internalplot(origstat, alldata, col="#377EB8", add=TRUE, seg=FALSE)
		#legend("topright", fill=c("#E41A1C", "#377EB8"), legend=c("Max T", "All")) 
		
	return(invisible(NULL))
}

print.seqnullcqi <- function(x, norm=TRUE, quant=0.95, digits=2, ...){
	cat("Parametric bootstrap cluster analysis validation\n")
	cat("Sequence analysis null model:", deparse(x$nullmodel), "\n")
	cat("Number of bootstraps:", x$R, "\n")
	cat("Clustering method:", ifelse(x$kmedoid, "PAM/K-Medoid", paste0("hclust with ", x$hclust.method)), "\n")
	cat("Seqdist arguments:", deparse(x$seqdist.args), "\n\n\n")

	alls <- as.data.frame(x$clustrange$stats)
	quants <- rep("", ncol(alls))
	names(quants) <- colnames(alls)
	for(ss in colnames(alls)){
		sumcqi <- normstatcqi(x, stat=ss, norm=norm)
		alls[, ss] <- as.character(round(sumcqi$origstat, digits=digits))
		borne <- as.character(round(confcqi(sumcqi$alldatamax, quant, x$R),  digits=digits))
		quants[ss] <- paste0("[", borne[1], "; ", borne[2],"]")
	}
	alls <- rbind(alls, rep("", length(quants)), quants)
	rownames(alls) <- c(rownames(x$clustrange$stats), "", paste("Null Max-T", quant, "interval"))
	print(alls, ...)
	return(invisible(alls))
}

plot.seqnullcqi <- function(x, stat, type=c("line", "density", "boxplot", "seqdplot"), quant=0.95, norm=TRUE, legendpos="topright", alpha=.2, ...){
	bcq <- x
	type <- type[1]
	if(type=="density"){
		return(plotncqdensity2(bcq, stat, quant=quant, norm=norm, maxt=TRUE,  ...))
	}
	if(type=="seqdplot"){
		return(seqdplot(bcq$seqdata,  ...))
	}
	
	cq <- bcq$clustrange
	kvals <- cq$kvals
	if(stat=="all"){
		alls <- list()
		for(ss in colnames(cq$stats)){
			nstat <- cq$stats[, ss]
			allstat <- rbind(nstat, bcq$stats[[ss]])
			for(i in seq_along(nstat)){
			  nstat[i] <- (nstat[i]-mean(allstat[, i]))/sd(allstat[, i])
			}
			alls[[ss]] <- nstat
		}
		
		plot(kvals, nstat, type="n", main="Standardized quality measure", ylab="Normalized quality measure", xlab="Number of clusters", ylim=range(unlist(alls)))
		for(ss in seq_along(alls)){
			lines(kvals, alls[[ss]], type="b", col=ss, ...)
		}
		
		legend(legendpos, fill=seq_along(alls), legend=names(alls))
		return(invisible(NULL))
	}
	origstat <- cq$stats[, stat]
	nullstat <- bcq$stats[[stat]]
	opar <- par(mfrow=c(1, 2))
	on.exit(par(opar))
	internalplot <- function(kvals, origstat, nullstat, main, ylab, ...){
		allstat <- rbind(origstat, nullstat)
		allrange <- range(allstat)
		alpha <- (1-quant)/2
		#calpha <- alpha+(alpha-1)/nrow(nullstat)		
		if(type!="boxplot"){
			plot(kvals, origstat, ylim=allrange, lwd=2, col="black", type="b", main=main, ylab=ylab, xlab="Number of clusters")
			
			for(i in 1:nrow(nullstat)){
				lines(kvals, nullstat[i,], col=gray(.5, alpha=alpha), lwd=1)
			}
			
			if(!is.null(quant)){
				minmax <- sapply(seq_along(kvals), function(x)quantile(nullstat[, x], c(alpha, 1-alpha))) 
				polygondata <- data.frame(x=c(kvals, rev(kvals), kvals[1]), y=c(minmax[2,], rev(minmax[1,]), minmax[2,1]))
				polygon(polygondata, col=adjustcolor("lightgray", alpha.f=0.15))  
			}
			lines(kvals, origstat, lwd=2, col="black", type="b")
		}else{
			nn <- as.vector(nullstat)
			kk <- rep(kvals, each=nrow(nullstat))
			boxplot(nn~kk, ylim=allrange, main=main, ylab=ylab, xlab="Number of clusters")
			#plot(kvals, origstat, ylim=allrange, lwd=2, col="black", type="b", main=main, ylab=ylab, xlab="Number of clusters")
			lines(seq_along(unique(kk)), origstat, lwd=2, col="black", type="b")
		}
		
		if(!is.null(quant)){
			threashold <- ifelse(stat=="HC", alpha, c(1-alpha))
			#overallq <- quantile(as.vector(nullstat), threashold)
			#abline(h=overallq, lty=3, col="#377EB8", lwd=2)
			if(stat=="HC"){
				overallmaxq <- quantile(apply(nullstat, 1, min), threashold)
			}else{
				overallmaxq <- quantile(apply(nullstat, 1, max), threashold)
			}
			abline(h=overallmaxq, lty=3, col="#E41A1C", lwd=2)
		}
	}
	internalplot(kvals, origstat, nullstat, main=paste("Raw", stat), ylab=stat)
	
	normstat <- origstat
	normnullstat <- nullstat 
	for(i in seq_along(normstat)){
		mn <- mean(nullstat[, i])
		sdn <- sd(nullstat[, i])
		normstat[i] <- (origstat[i]-mn)/sdn
		normnullstat[, i] <- (normnullstat[, i]-mn)/sdn
	}
	internalplot(kvals, normstat, normnullstat, main=paste("Standardized", stat), ylab=paste("Standardized", stat))
	##plot(kvals, nstat, type="b", main=, ylab="Normalized quality measure", xlab="Number of clusters")

}

seqnullcqi <- function(seqdata, clustrange, R, model=c("combined", "duration", "sequencing", "stateindep", "Markov", "userpos"),
						seqdist.args=list(), kmedoid=FALSE, hclust.method="ward.D", parallel=FALSE, progressbar=FALSE, ...){
	if(!inherits(clustrange, "clustrange")){
		stop(" [!] Original cluster quality measures should be provided as a clustrange object. See ?as.clustrange().\n")
	}
	if (!inherits(seqdata,"stslist")){
		stop(" [!] seqdata should be a sequence object, see seqdef function to create one")
	}
	if(missing(R)){
			stop(" [!] An R value should be specified.")
	}
	ncluster <- max(clustrange$kvals)
	if(!all.equal(clustrange$kvals, 2:ncluster)){
		stop(" [!] Original cluster quality measures should be computed for the number of groups from 2 to ncluster\n")
	}
	extractStat <- function(bcq, stat="ASW"){
		bb <- t(sapply(bcq, function(x) x[, stat]))
		colnames(bb) <- rownames(bcq[[1]])
		return(bb)
	}
	nc <- list()
	allseq <- list()
	oldseqdist.args <- seqdist.args
	if(parallel){
		oplan <- plan(multisession)
		on.exit(plan(oplan), add=TRUE)
	}
	#if(ncores>1){
		#cl <- parallel::makeCluster(ncores)
		#on.exit(parallel::stopCluster(cl))
		#doSNOW::registerDoSNOW(cl)
		#pb <- progress::progress_bar$new(format = "(:spin) [:bar] :percent | Elapsed: :elapsed | ETA: :eta", total = R)
		  #pb <- txtProgressBar(max = R, style = 3)
		  #progress <- function(n) setTxtProgressBar(pb, n)
		#opts <- list(progress = function(n) pb$tick())
		#`%dopar%` <- foreach::`%dopar%`
	if(progressbar){	
		if (requireNamespace("progress", quietly = TRUE)) {
				old_handlers <- handlers(handler_progress(format   = "(:spin) [:bar] :percent | Elapsed: :elapsed | ETA: :eta | :message"))
				if(!is.null(old_handlers)){
					on.exit(handlers(old_handlers), add = TRUE)
				}
			}else{
				message(" [>] Install the progress package to see estimated remaining time.")
			}
			oldglobal <- handlers(global=TRUE)
			if(!is.null(oldglobal)){
				on.exit(handlers(global=oldglobal), add = TRUE)
			}
	}
		p <- progressor(R)
		#parObject <- foreach::foreach(loop=1:R,  .packages = c('TraMineR', 'WeightedCluster'), .options.future = list(seed = TRUE)) %dofuture% {#on stocke chaque
		parObject <- foreach(loop=1:R, .options.future = list(seed = TRUE)) %dofuture% {#on stocke chaque
			suppressMessages(ss <- seqnull(seqdata, model=model, ...))
			sarg <- seqdist.args
			sarg$seqdata <- ss
			suppressMessages(diss <- do.call(seqdist, sarg))
			if(kmedoid){
				nc <- wcKMedRange(diss=diss, kvals=2:ncluster)$stats
			}else{
				hc <- hclust(as.dist(diss), method="ward.D")
				nc <- as.clustrange(hc, diss=diss, ncluster=ncluster)$stats
			}
			rm(diss)
			gc()
			#p(message=sprintf("Iteration %d", loop))
			p()
			list(allseq=ss, nc=nc)
		}
		
		allseq <- lapply(parObject, function(x)x$allseq)
		nc <- lapply(parObject, function(x)x$nc)
		
	# }
	# else{
		
		# for(i in 1:R){
			# suppressMessages(ss <- seqnull(seqdata, model=model, ...))
			# seqdist.args$seqdata <- ss
			# allseq[[i]] <- ss
			# suppressMessages(diss <- do.call(seqdist, seqdist.args))
			# if(kmedoid){
				# nc[[i]] <- wcKMedRange(diss=diss, kvals=2:ncluster)$stats
			# }else{
				# hc <- hclust(as.dist(diss), method="ward.D")
				# nc[[i]] <- as.clustrange(hc, diss=diss, ncluster=ncluster)$stats
			# }
		# }
	# }
	stnames <- colnames(nc[[1]])
	stats <- list()
	for(st in stnames){
		stats[[st]] <- extractStat(nc, st)
	}
	allseq <- do.call(rbind, allseq)
	bcq <- list(seqdata=allseq, stats=stats, clustrange=clustrange, R=R, kmedoid=kmedoid, hclust.method=hclust.method, seqdist.args=oldseqdist.args, nullmodel=list(model=model, ...))
	class(bcq) <- "seqnullcqi"
	return(bcq)
}

