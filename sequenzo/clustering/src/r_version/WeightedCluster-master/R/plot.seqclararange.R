

plot.seqclararange <- function(x, stat="CQI", type = "o", main = NULL, 
								xlab = "Number of clusters", ylab = stat, col ="blue", legend.pos="topright",
								pch = 19, norm="none",  ...){
								
	if(!inherits(x, "seqclararange")){
		stop(" [!] x should be a seqclararange object.")
	}
	if(stat %in% c("stability", "stabmean") && any(is.na(x$clara[[1]]$arimatrix))){
		stop(" [!] stability evaluation is not available for this object. Use stability=TRUE in seqclararange to compute stability values.")
	}
	
  plotlines <- function(y){
    if(!is.matrix(y)){
      plot(x$kvals, y, type=type, main=main, xlab=xlab, ylab=ylab, col=col, pch=pch, ...)
    }else{
      ylim <- range(unlist(y), finite=TRUE)
      xlim <- range(unlist(x$kvals), finite=TRUE)
      cols <- brewer.pal(ncol(y)+1, "Set3")[-2]
      plot(xlim, ylim, xlim=xlim, ylim=ylim, type="n", xlab=xlab, ylab=ylab, main=main)
			for(tt in 1:ncol(y)){
				lines(x$kvals, y[, tt], col=cols[tt], type=type, ...)
			}
      legend(legend.pos, fill=cols, legend=colnames(y))
    }
  }
  
  
  
  if(stat %in% c("CQI", "CQImv")){
    if(is.null(main)){
      main <- "Cluster Quality Indices"
    }
	x$stats <- x$stats[, c("PBM", "DB", "XB", "AMS")]
	PBMK <- x$stats[, "PBM"]*sqrt(x$kvals)
	#PBMK2 <- x$stats[, "PBM"]*x$kvals^2
	x$stats <- cbind(x$stats, PBMK=PBMK)
	if(stat=="CQImv"){
		MV <- rank(x$stats[, "SASW"])+rank(x$stats[, "PBM"])-rank(x$stats[, "DB"])-rank(x$stats[, "XB"])
		MVs <- scale(x$stats[, "SASW"])+scale(x$stats[, "PBM"])-scale(x$stats[, "DB"])-scale(x$stats[, "XB"])
		x$stats <- cbind(x$stats, MV=MV, MVs=MVs[, 1])
	}
	
    y <- normalize.values.all(x$stats, norm = norm)
    
	plotlines(y)
  }else if(stat=="ari.boxplot"){
	  ari <- lapply(seq_along(x$kvals), function(i) {
	    ari <- x$clara[[i]]$arimatrix
	    if(is.matrix(ari)) ari <- ari[, x$clara[[1]]$iteration]
	    data.frame(kvals=x$kvals[i], ari=ari)
	   })
		ari <- do.call(rbind, ari)
		boxplot(ari~kvals, data=ari, main=main, xlab=xlab, ylab=ylab, col=col,...)
  }else if(stat == "stability"){
    recovery <- function(min.ari, aricol=1){
      sapply(seq_along(x$kvals), function(i){
  			ari <- x$clara[[i]]$arimatrix[, aricol]
			ret <- sum(ari >= min.ari)
  			return(ret)
  		})
    }
	  y <- cbind(recovery(0.9), recovery(0.8), recovery(0.7), recovery(0.5, 2), recovery(2/3, 2), recovery(0.8, 2))
	  colnames(y) <- c("Strong Recovery(ARI > 0.9)", "Good Recovery (ARI > 0.8)", "Weak Recovery (ARI > 0.7)", "Minimal Recovery (JC > 0.5)", "Weak Recovery (JC > 0.66)", "Strong Recovery (JC > 0.8)")
	  plotlines(y)
	
	}else if(stat == "stabmean"){
		avg <- function(quant, aricol){
		  sapply(seq_along(x$kvals), function(i){
				cond <- x$clara[[i]]$iter.objective <= quantile(x$clara[[i]]$iter.objective, quant)
				ari <- x$clara[[i]]$arimatrix[cond, aricol]
				ret <- mean(ari)
				return(ret)
			})
		}
	  y <- cbind(avg(1, 1), avg(0.2, 1), avg(1, 2), avg(0.2, 2))
	  
	  colnames(y) <- c("Average ARI", "Trimmed average ARI (20%)", "Average JC",  "Trimmed average JC (20%)")
	  plotlines(y)
	}
}

