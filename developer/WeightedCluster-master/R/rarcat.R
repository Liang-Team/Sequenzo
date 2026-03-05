
rarcat <- function(formula, data, diss, 
                   robust=TRUE, R=500, 
                   kmedoid=FALSE, hclust.method="ward.D", 
                   fixed=FALSE, ncluster=10, cqi="HC",
                   parallel=FALSE, progressbar=FALSE,
                   fisher.transform=FALSE, 
				   lmerCtrl=lme4::lmerControl()) {
  
  
  # Ensure dissimilarity matrix and dataset have the same size
  if(nrow(data) != nrow(diss)) {
	stop(" [!] Please verify that the dissimilarity matrix corresponds to the dataset")
  }
  if(length(ncluster) > 1) {
	stop(" [!] Please give a single number as kcluster (see documentation)")
  }
  if(!(cqi %in% c("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", 
                   "R2sq", "HC"))) {
    stop("Incorrect evaluation measure")
  }
  if(ncluster < 2) stop("At least two clusters required")
  
  # Return object
  ret <- list(arguments = list(formula=formula, robust=robust, R=R,  
                   kmedoid=kmedoid, hclust.method=hclust.method, 
                   fixed=fixed, ncluster=ncluster, cqi=cqi, fisher.transform=fisher.transform))
  
  #Update the formula with membership as dependent
  # Extract the right-hand side (RHS)
  rhs <- deparse(formula[[3]])  
  lhs <- deparse(formula[[2]])  
  modelDF <- model.frame(formula, data)
  ret$formula <- formula
  # Update formula, ensuring we keep the original environment of the formula
  membershipFormula <- paste("membership~", rhs)

  clustering <- modelDF[, lhs]
   #<- paste("membership ~", paste(colnames(modelDF)[-1], collapse = " + "))
  ret$AMElist <- list()
  
  ret$clusterNames <- as.character(unique(clustering))
  
  for(i in ret$clusterNames) {
    modelDF$membership <- clustering == i
    mod <- glm(membershipFormula, modelDF, family = "binomial")
    ret$AMElist[[i]] <- summary(margins::margins(mod))
    #print(tmp)
	
  }
  factorName <- unique(c(sapply(ret$AMElist, function(x) as.character(x$factor))))
  # Run the bootstrap procedure if asked
  #originalAMEmatrix <- createAMEmatrix(clustering, ret$AMElist, seq_len(nrow(data)))
  
  
  #############################################
  ## This is where the bootstrap takes place
  #############################################
  
  if(robust) {
    
	## Parallel initialization
	if (parallel) {
		message(" [>] Initializing parallel processing...", appendLF = FALSE)
		oplan <- plan(multisession)
		on.exit(plan(oplan), add = TRUE)
		message(" OK.")
	}
	if (progressbar) {
		if (requireNamespace("progress", quietly = TRUE)) {
		  old_handlers <- handlers(handler_progress(format = "(:spin) [:bar] :percent | Elapsed: :elapsed | ETA: :eta | :message"))
		  if (!is.null(old_handlers)) {
			on.exit(handlers(old_handlers), add = TRUE)
		  }
		} else {
		  message(" [>] Install the progress package to see estimated remaining time.")
		}
		oldglobal <- handlers(global = TRUE)
		if (!is.null(oldglobal)) {
		  on.exit(handlers(global = oldglobal), add = TRUE)
		}
	}
	p <- progressor(R)

	  ## Memory clean up before parallel computing
	gc()
	message(" [>] Starting bootstraps...\n")
	  
	## Random component made at the beginning relies on current seed, no parallel issues.
	## Should we add the original indices in the first bootstraps ?
	boot_indices <- replicate(R, sample(seq_len(nrow(data)), replace = TRUE), simplify = FALSE)
	#boot_indices[[1]] <- seq_len(nrow(data))
	amemat_internal <- getFromNamespace("amemat", "WeightedCluster")
	
	## Parallelized bootstrapping loop 	
	res_list <- foreach(i = seq_len(R), .options.future = list(seed = TRUE, 
							globals=c("diss", "boot_indices", "formula", "modelDF", "kmedoid", "hclust.method",
									"fixed", "ncluster", "cqi",  "p", "amemat_internal"),  # reproducible RNG
						packages = c("WeightedCluster", "stats", "progressr", "margins"))) %dofuture% {

		  # Subsample indices
		indices <- boot_indices[[i]]
		  
		# Bootstrap
		ame <- amemat_internal(diss = diss,
						indices=indices,
						formula = formula,
						modelDF = modelDF,
						kmedoid = kmedoid, 
						hclust.method=hclust.method,
						fixed = fixed, 
						ncluster=ncluster, 
						cqi=cqi, 
						debug=FALSE)
		p()
		ame
	}
	message(" [>] Finished bootstraps.\n")
	  
	#res <- boot::boot(diss, amemat, R, formula=formula, data1=data, 
	#					kmedoid = kmedoid, hclust.method=hclust.method, cqi=cqi, ncluster=ncluster, fixed=fixed)
	
	message(" [>] Pooling results...", appendLF = FALSE)
	cluster.solution <- sapply(res_list, function(x) (x)[, "clustering"]) 
	optimal.number <- apply(cluster.solution, 2, function(x) length(unique(na.omit(x))))
	effects <- list()
	errors <- list()
	ids <- seq_along(clustering)
	
	## Pooled coefficients and standard errors
	pooled.ame <- matrix(NA, nrow=length(factorName), ncol=length(ret$clusterNames), dimnames=list(factorName, ret$clusterNames))
	standard.error <- pooled.ame
	bootstrap.stddev <- pooled.ame
	observation.stddev <- pooled.ame

	## Indivudal &bootstrap ranef
	bootstrap.ranef <- matrix(NA, nrow=R, ncol=length(factorName), dimnames=list(as.character(seq_len(R)), factorName))
	observation.ranef <- matrix(NA, nrow=nrow(data), ncol=length(factorName), dimnames=list(as.character(ids), factorName))
	observation.stdranef <- matrix(0, nrow=nrow(data), ncol=length(factorName), dimnames=list(as.character(ids), factorName))
	
	for (ff in factorName) {
		effects[[ff]] <- sapply(res_list, function(x) (x)[, ff]) 
		errors[[ff]] <- sapply(res_list, function(x) (x)[, paste(ff, "SE")])
		for(cc in ret$clusterNames){
			clustcond <- clustering == cc
			ame <- effects[[ff]][clustcond,]
			error <- errors[[ff]][clustcond,]
			# Prepare the data to input in the multilevel model
			prep <- data.frame(bootstrap = rep(1:ncol(ame), nrow(ame)),
							 id = rep(ids[clustcond], each=ncol(ame)),
							 ame = c(t(ame)), 
							 sterror = c(t(error)))
			prep <- subset(prep, !is.na(ame))
			# Weights based on the standard errors
			prep$weight <- 1/prep$sterror^2
			prep$stweight <- prep$weight/mean(prep$weight)
			# Atanh for Fisher's z-transformation
			if(fisher.transform) prep$ame <- atanh(prep$ame)
		  
			rob <- suppressMessages(lme4::lmer(ame ~ (1|id) + (1|bootstrap), 
											 weights = prep$stweight, data = prep,  control =lmerCtrl))
			output <- summary(rob)

			if(fisher.transform) {
				#tanh for inverse Fisher's z-transformation
				output$coefficients[1] <- tanh(output$coefficients[1])
				output$coefficients[2] <- output$coefficients[2] * (1 - tanh(output$coefficients[1])^2)
				output$varcor$bootstrap <- tanh(as.numeric(output$varcor$bootstrap))
				output$varcor$id <- tanh(as.numeric(output$varcor$id))
			}
			pooled.ame[ff, cc] <- output$coefficients[1]
			standard.error[ff, cc] <- output$coefficients[2]
			bootstrap.stddev[ff, cc] <- sqrt(as.numeric(output$varcor$bootstrap))
			observation.stddev[ff, cc] <- sqrt(as.numeric(output$varcor$id))
		  
			#bootstrap.ranef[clustcond, ff] <- lme4::ranef(rob)$bootstrap[,1]
			observation.ranef[clustcond, ff] <- lme4::ranef(rob)$id[,1]
			if(observation.stddev[ff, cc]!=0){
				observation.stdranef[clustcond, ff] <- observation.ranef[clustcond, ff]/observation.stddev[ff, cc]
			}
			#else{
				#warning(" Observation-level standard deviation estimated at zero for covariate ", ff, " cluster ", cc) 
			#}
		}
	  }
	  message("OK.")
	  ret$bootout <- list(cluster.solution=cluster.solution, optimal.number=optimal.number, 
						effects=effects, errors=errors)
		ret$pooled.ame <- pooled.ame
		ret$standard.error <- standard.error
		ret$bootstrap.stddev <- bootstrap.stddev
		ret$observation.stddev <- observation.stddev
		## Indivudal &bootstrap ranef
		ret$observation.ranef <- observation.ranef
		ret$observation.stdranef <- observation.stdranef
		ret$cluster.solution <- cluster.solution
		ret$optimal.number <- optimal.number

	}
	
	ret$factorName <- factorName
	ret$clustering <- factor(clustering)
	class(ret) <- "rarcat"
	return(ret)
}


print.rarcat <- function(x, conf.level=0.95, single.row = FALSE, digits = 3, ...) {
	
	internal_print <- function(coef, upper, lower){
		
		if(single.row){
			rnames <- x$factorName
		}else{
			rnames <- rep("", 2*length(x$factorName))
			crow <- 2*seq_along(x$factorName)-1
			cirow <- crow +1
			rnames[crow] <- x$factorName
			
		}
		tab <- matrix("",
				  nrow = length(rnames),
				  ncol = length(x$clusterNames),
				  dimnames = list(rnames, paste("Cluster", x$clusterNames)))
		for(cc in seq_along(x$clusterNames)){
			txtconfint <- sprintf("(%s; %s)", format(upper[, cc], digits=digits), format(lower[, cc], digits=digits))
			txtcoef <- format(coef[, cc], digits=digits)
			if(single.row){
				tab[, cc] <- sprintf("%s %s", txtcoef, txtconfint)
			}else{
				tab[crow, cc] <- txtcoef
				tab[cirow, cc] <- txtconfint
			}
		}
		print(tab, quote = FALSE, ...)
		return(tab)
	}
	alpha <- (1-conf.level)
	z_test <- qnorm(1 - alpha/2)
	## Compute coef and conf interval for base analysis.
	coefmat <- matrix(NA,
				  nrow = length(x$factorName),
				  ncol = length(x$clusterNames))
	uppermat <- coefmat
	lowermat <- coefmat
		
	for(i in seq_along(x$clusterNames)) {
		tmp <- x$AMElist[[x$clusterNames[i]]]
		coefmat[,i] <- tmp$AME
		lowermat[, i] <- tmp$AME - z_test*tmp$SE
		uppermat[, i] <- tmp$AME + z_test*tmp$SE
    }
	cat("\n NAIVE ESTIMATES\n Average Marginal Effect (AME) to be in a given cluster instead of any others\n")
	
	base <- internal_print(coefmat, lowermat, uppermat)
	
	if(x$arguments$robust){
		coefmat <- x$pooled.ame
		var <- qt(1 - alpha/2, x$arguments$R - 2)* sqrt(x$standard.error^2 + x$bootstrap.stddev^2)
		uppermat <- coefmat + var
		lowermat <- coefmat - var
		cat("\n ROBUST ESTIMATES (RARCAT)\n Average Marginal Effect (AME) to be in a given cluster instead of any others\n\n")
		cat(" Number of bootstraps: ", x$arguments$R, " with ", ifelse(x$arguments$fixed, "fixed", "varying"), " number of clusters\n")
		if(!x$arguments$fixed){
			cat("Distribution of the number of clusters across bootstraps\n")
			print(table(x$optimal.number))
			cat("\n")
		}
		rarcat <- internal_print(coefmat, lowermat, uppermat)

	}else{
		cat("\n Robust estimates not estimated\n")
		rarcat <- NULL
	}
	
    invisible(list(base, rarcat))
}

summary.rarcat <- function(object, ...) {
	
	cat("\nRARCAT Diagnostics\nDistribution of standardized observation random intercept\n")
	for(cc in colnames(object$observation.stdranef)){
		cat("\n", cc, "\n")
		print(table(cut(object$observation.stdranef[, cc], breaks=c(-Inf, -2, -1, 1, 2, Inf), labels=c("< -2", "-2....-1", "-1...1", "1...2", ">2"))), ...)
		cat("\n")
	}
	
	cat("\nStandard errors\n")
	print(object$standard.error, quote=FALSE, ...)
	
}

plot.rarcat <- function(x, what="AME", covar=x$factorName[1], pooled.ame=TRUE, naive.ame=TRUE,  
						 with.legend=TRUE, legend.prop=NA, rows=NA, 
						cols=NA, main=NULL, xlab=paste(covar, "Average Marginal Effect"),  xlim=NULL, conf.level=0.95, ...){
	if(!x$arguments$robust){
		stop(" [!] Plot is only available for robust/RARCAT analysis")
	}
	savepar <- par(no.readonly = TRUE)
	on.exit(par(savepar))
	if(what=="AME"){
		alpha <- (1-conf.level)
		z_test <- qnorm(1 - alpha/2)
		eff <- x$bootout$effects[[covar]]
		if(is.null(xlim)) xlim <- range(c(sapply(x$AMElist, function(x) x[x$factor==covar, "AME"]), eff), na.rm=TRUE)
		lout <- TraMineRInternalLayout(length(x$clusterNames), rows, cols, with.legend=with.legend, axes="all", legend.prop)
		
		layout(lout$laymat, heights=lout$heights, widths=lout$widths)
		for(cc in seq_along(x$clusterNames)){
			ccmain <- paste(ifelse(is.null(main), "Cluster", main), x$clusterNames[cc], sep=" - ")
			hh <- hist(eff[x$clustering==x$clusterNames[cc], ], xlim=xlim, xlab=xlab, main=ccmain, ...)
			if(pooled.ame){
				pp <- x$pooled.ame[covar, x$clusterNames[cc]]
				var <- qt(1 - alpha/2, x$arguments$R - 2)* sqrt(x$standard.error[covar, x$clusterNames[cc]]^2 + x$bootstrap.stddev[covar, x$clusterNames[cc]]^2)		
				lower <- pp - var
				upper <- pp + var
				abline(v=pp, col="red")
				polygon(c(lower, lower, upper, upper), c(0, max(hh$counts), max(hh$counts), 0), col = adjustcolor("red", alpha.f=.3), border = NA)
				
				max(hh$counts)
			}
			if(naive.ame){
				tmp <- x$AMElist[[x$clusterNames[cc]]]
				cond <- tmp$factor==covar
				lower <- tmp$AME[cond] - z_test*tmp$SE[cond]
				upper <- tmp$AME[cond] + z_test*tmp$SE[cond]
				abline(v=tmp$AME[cond], col="blue")
				polygon(c(lower, lower, upper, upper), c(0, max(hh$counts), max(hh$counts), 0), col = adjustcolor("blue", alpha.f=.3), border = NA)
			}
		}
		 
		plot(0, type = "n", axes = FALSE, xlab = "", ylab = "")
		legend("topleft", legend=c("Naive estimate (with CI)", "RARCAT estimate (with CI)"), fill=c("blue", "red"))
    }else if(what=="ranef"){
		if(is.null(xlim)) xlim <- range(c(x$observation.stdranef), na.rm=TRUE)
		lout <- TraMineRInternalLayout(length(x$clusterNames), rows, cols, with.legend=FALSE, axes="all", legend.prop)
		
		layout(lout$laymat, heights=lout$heights, widths=lout$widths)
		for(cc in seq_along(x$clusterNames)){
			ccmain <- paste(ifelse(is.null(main), "Cluster", main), x$clusterNames[cc], sep=" - ")
			hh <- hist(x$observation.stdranef[x$clustering==x$clusterNames[cc], covar], xlim=xlim, xlab=xlab, main=ccmain, ...)
		}
	}	

		
}




	# ret$bootout <- regressboot(formula, data, diss, clustering=clustering,
								# factorName=factorName, R=R, kmedoid = kmedoid, 
								# hclust.method=hclust.method, fixed=fixed, 
								# ncluster=ncluster, cqi=cqi, parallel=parallel, 
								# progressbar=progressbar)
	# ret$bplist <- list()
	# for(i in ret$clusterNames) {
		# ret$bplist[[i]] <- list()
		# for(fn in factorName) {
			#Run the pooling function for each combination of cluster and covariate
			# ret$bplist[[i]][[fn]] <- bootpool(ret$bootout, pooledData, i, fn, fisher.transform)
		# }
	# }
  # }
  