
amemat <- function(diss, indices, formula, modelDF, kmedoid = FALSE, hclust.method="ward.D",
                   fixed = FALSE, ncluster=10, cqi="HC", debug=FALSE) {
	# Bootstrap dissimilarities
  d <- diss[indices,indices]
  
  
  ## Clustering the boostrapped sample
  if(fixed||ncluster==2) {
	if(kmedoid) {
		clustering <- wcKMedoids(diss=d, k=ncluster, cluster.only=TRUE)
	} else {
		tree <- fastcluster::hclust(as.dist(d), method = hclust.method)
		clustering <- cutree(tree, k=ncluster)
	}
  }else{ ## Find the best number of group
	if(kmedoid) {
		quality <- wcKMedRange(d, kvals=2:ncluster)
	} else {
		tree <- fastcluster::hclust(as.dist(d), method = hclust.method)
		quality <- as.clustrange(tree, diss=d, ncluster=ncluster)
	}
	if(cqi=="HC"){
		best <- which.min(quality$stats[, cqi])
	}else{
		best <- which.max(quality$stats[, cqi])
	}
	clustering <- quality$clustering[, row.names(quality$stats)[best]]
  }
  clustering <- as.numeric(factor(clustering))
  ## Association study
  # Bootstrap covariates
  
  rhs <- deparse(formula[[3]])  
  lhs <- deparse(formula[[2]])  
  # Update formula, ensuring we keep the original environment of the formula
  membershipFormula <- paste("membership~", rhs)
  
  bdata <- modelDF[indices,]
  # Recreate ids for safety
  # Changing bootstrap clustering
  bdata[,lhs] <- clustering
  
  # Preallocate list
  
  
  #modelFormula <- paste("membership ~", paste(colnames(modelDF)[-1], collapse = " + "))
  output_list <- list()

	ndupl <-  !duplicated(indices)
	
  for(i in unique(clustering)) {
    
    if(debug) print(i)
    
    bdata$membership <- clustering == i
    mod <- glm(membershipFormula, bdata, family = "binomial")
    tmp <- summary(margins::margins(mod))
	ids <- indices[clustering == i & ndupl]
	effects <- matrix(tmp$AME, nrow = length(ids), ncol = nrow(tmp), byrow = T)
	colnames(effects) <- tmp$factor
	errors <- matrix(tmp$SE, nrow = length(ids), ncol = nrow(tmp), byrow = T)
	colnames(errors) <- paste(tmp$factor, "SE")
	
	output_list[[i]] <- cbind(ids, clustering=i, effects, errors)
  }
  
	output <- do.call(rbind, output_list)
	m <- match(seq_len(length(clustering)), output[, 1])
	return(as.matrix(output[m, -1]))
#  m <- match(seq_len(nrow(data)), output[, "id"])
  # With all individuals (and missing observations depending on the sample)
  #output <- dplyr::left_join(data.frame(id = ), output, by = "id")
 # return(c(as.matrix(output[m, -1])))
}

