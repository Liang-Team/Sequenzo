


davies_bouldin_internal <- function(diss, clustering, medoids, p=1, weights=NULL, medoidclust=FALSE){

  list_diam <- numeric(length(medoids))
  if(is.null(weights)){
	weights <- rep(1, nrow(diss))
  }

  for(i in seq_along(medoids)){ #on stocke chaque sample
    #AMELIORATION CAR DISS RETURNED IN OBJ
	medi <- ifelse(medoidclust, medoids[i], i)
	cond <- clustering == medi
    list_diam[i] <- (sum(weights[cond]*diss[cond, i]**p)/sum(weights[cond]))**(1/p)
  }
  maximum <- rep(0,length(medoids))
  for(i in seq_along(medoids)){ #pour chaque sous-groupes

    maximum2 <- (list_diam[i] + list_diam)/diss[medoids[i],]
    maximum[i] <- max(maximum2[is.finite(maximum2)]) ## ensure values for "same" medoids
  }

  final_db <- mean(maximum)
  # db_evolution <- cumsum(maximum) / c(1:length(seq_obj$id.med))
  ret <- list(db=final_db, per.cluster=maximum)
  return(ret)

}


fuzzy_davies_bouldin_internal <- function(diss, memb, medoids, weights=NULL){

  list_diam <- numeric(length(medoids))
  if(is.null(weights)){
		weights <- rep(1, nrow(diss))
  }

  n <- sum(weights)
  mw <- memb*weights
  list_diam <- colSums(mw*diss)/colSums(mw)
  
  maximum <- rep(0,length(medoids))
  for(i in seq_along(medoids)){ #pour chaque sous-groupes

    maximum2 <- (list_diam[i] + list_diam)/diss[medoids[i],]
    maximum[i] <- max(maximum2[is.finite(maximum2)]) ## ensure values for "same" medoids
  }

  final_db <- mean(maximum)
  # db_evolution <- cumsum(maximum) / c(1:length(seq_obj$id.med))
  ret <- list(db=final_db, per.cluster=maximum)
  return(ret)

}


adjpbm_internal <- function(diss, clustering, medoids, p=1, weights=NULL, medoidclust=FALSE){
  
  
  if(is.null(weights)){
    weights <- rep(1, nrow(diss))
  }
  internaldist <- sapply(seq_along(medoids), FUN=function(i){
    medi <- ifelse(medoidclust, medoids[i], i)
    cond <- clustering == medi
    (sum(weights[cond]*diss[cond, i]**p)/sum(weights[cond]))**(1/p)
  })
  separation <- min(as.dist(diss[medoids, ]), na.rm=TRUE)
  pbm <- (1/length(medoids)) *(separation/sum(internaldist)) 
  return(pbm)
  
}


