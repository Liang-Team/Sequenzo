## seqrf  computes values for relative frequency plot

dissrf <- function(diss, k=NULL, sortv="mds", weights=NULL,
						#ylab=NA, yaxis=FALSE, main=NULL, which.plot="both",
                        grp.meth = "prop", squared = FALSE, pow = NULL){
	
	return(dissrf_internal(diss, k=k, sortv=sortv, weights=weights,
						#ylab=ylab, yaxis=yaxis, main=main, which.plot=which.plot,
                        grp.meth = grp.meth, squared = squared, pow = pow
                        )
           )
}

dissrf_internal <- function(diss, k=NULL, sortv="mds", weights=NULL,
                            use.hclust=FALSE, hclust_method="ward.D", #use.quantile=FALSE,
                            #ylab=NA, yaxis=FALSE, main=NULL, which.plot="both",
                            grp.meth = "prop", squared = FALSE, pow = NULL){

  if (inherits(diss, "dist")) diss <- as.matrix(diss)
  ncase <- dim(diss)[1]

  if (is.null(pow)){
    pow <- if (squared) 2 else 1
  }
  mdspow <- 2^squared

  if (!is.null(sortv)) {
    if (length(sortv)==1 && sortv != "mds")
        stop(call.=FALSE, ' Only length one value allowed in dissrf is "mds"')
    else if (length(sortv) != nrow(diss) & length(sortv) != 1)
        stop(" length of sortv not equal to nrow(diss)")
  }

  ## normalizing weights
  if (is.null(weights)){
    weights <- rep(1,ncase)
    weighted <- FALSE
  } else {
    if (grp.meth != "prop") {
      msg.stop(paste0("Selected grp.meth '",grp.meth,"' does not apply to weighted data! Use grp.meth='prop'."))
    }
    #weights <- ncase * weights/sum(weights)
    weighted <- TRUE
    #grp.meth <- "prop"
  }

  wsum <- sum(weights)
  if (is.null(k)) k <- min(floor(wsum/10),100)
  message(" [>] Using k=", k, " frequency groups with grp.meth='",grp.meth,"'")

  #Extract medoid, possibly weighted
  gmedoid.index <- disscenter(diss, medoids.index="first", weights=weights, squared=squared)
  gmedoid.dist <-diss[, gmedoid.index] #Extract distance to general medoid
  sum.gmedoid.dist <- sum(gmedoid.dist^pow)


  ##Vector where distance to k medoid will be stored
  kmedoid.dist <- rep(0, ncase)
  #index of the k-medoid for each sequence
  kmedoid.index <- rep(0, ncase)
  #calculate qij - distance to frequency group specific medoid within frequency group

  if(is.null(sortv))
    sortv <- 1:nrow(diss)
  if(length(sortv)==1 && sortv=="mds" && !use.hclust){
    if (weighted)
        sortv <- wcmdscale(diss^mdspow, k = 1, w=weights)
    else
        sortv <- cmdscale(diss^mdspow, k = 1)
  }
  ## sort order
  sortorder <- order(sortv)
  cumweights <- cumsum(weights[sortorder])


  ## Grouping sequences and assigning group membership (mdsk)
  if (!(grp.meth %in% c("first", "random","prop")))
    stop(" grp.meth must be one of 'first', 'random', 'prop' ")
  ## TO DO  Taking weights into account ##

  dist.list <- list()
  index.list <- list()
  weights.list <- list()
  if(grp.meth=="prop"){ ## New method by Gilbert that can handle weights
    gsize <- wsum/k
    sumwdist <- rep(NA,k)
    wg <- matrix(rep(0,5*k),ncol=5)
    colnames(wg) <- c("start","end","wg.first","wg.last","medoid.idx")

    wg[1,1] <- 1
    wg[1,3] <- min(weights[sortorder][1],gsize)
    wg[k,2] <- ncase
    wg[k,4] <- weights[sortorder][ncase]
    wtemp <- wg[1,3]
    for (i in 1:k) {
      if (i<k) {
        i2 <- max(0,which(cumweights <= i*gsize)) + 1
        i2 <- min(i2,ncase)
        wg[i+1,1] <- wg[i,2] <- i2
        ## part of weight of first element of the group
        ##  needs special handling when weight > gsize
        wg[i+1,3] <-   min(cumweights[wg[i,2]] - i*gsize, gsize)
        if (wg[i+1,1] == wg[i,1]) {
          wtemp <- wtemp - wg[i+1,3]
        } else {
          wtemp <- weights[sortorder][wg[i+1,1]]
        }
        if (wg[i,2]>(wg[i,1]+1)){
          wgsum <- sum(weights[sortorder][(wg[i,1]+1):(wg[i,2]-1)])
        } else
          wgsum <- 0
        if (wg[i,2] > wg[i,1]) {
          wgsum <- wgsum + wg[i,3]
          wg[i,4] <- min(wtemp - wg[i+1,3], gsize - wgsum )
        } else
          wg[i,4] <- 0
      }

      ind <- sortorder[wg[i,1]:wg[i,2]]
      wind <- weights[ind]
      wind[length(ind)] <- wg[i,4]
      wind[1] <- wg[i,3]
      #wg[i,5] <- sum(wind) ## sum(wind) is gsize for all i

      weights.list[[i]] <- wind
      index.list[[i]] <- ind

      if (length(ind)==1){
        wg[i,5] <- ind
        dist.list[[i]] <- 0
        sumwdist[i] <- 0
      } else {
        dd <- diss[ind, ind]
        ##Identify medoid
        kmed <- disscenter(dd, medoids.index="first", weights=wind, squared=squared)
        wg[i,5] <- ind[kmed]

        ##Distance to medoid for each seq
        dist.list[[i]] <- dd[, kmed]

        ## sum of weighted (squared) distances to group medoid
        sumwdist[i] <- sum(weights.list[[i]]*dist.list[[i]]^pow)
      }
    }
    R2 <- 1 - sum(sumwdist)/sum.gmedoid.dist
    med.names <- rownames(diss)[wg[,5]]
  }
  else { ## original method by Tim,  grp.meth != "prop"
    if(!is.null(sortv)){
      ## sorted position
      ##gr## here we use sortorder instead
      ##sortpos <- rank(sortv, ties.method = "random")
      ng <- wsum %/% k
      r  <- wsum %% k

      n.per.group <- rep(ng, k)
      if(r>0){
        #n.per.group[order(runif(r))] <- ng+1
        ##gr 23.05.22: order(runif(r)) produces random order of 1:r
        ##    therefore above makes first r groups one unit larger
        if(grp.meth=="first"){
          n.per.group[1:r] <- ng + 1
        }
        else {
          ##   random selection fo r groups
          n.per.group[sample(1:k,r)] <- ng+1
        }
      }
      ##mdsk <- rep(1:k, n.per.group)
      ##mdsk <- mdsk[sortpos]
      mdsk <- vector("numeric", length=ncase)
      mdsk[sortorder] <- rep(1:k, n.per.group)



    }else{ ## using clusters
      hh <- hclust(as.dist(diss), method=hclust_method)
      mdsk <- factor(cutree(hh, k))
      medoids <- disscenter(diss, group=mdsk, medoids.index="first", squared=squared)
      medoids <- medoids[levels(mdsk)]
      #ww <- xtabs(~mdsk)
      ## sorting medoids and redefining group number accordingly
      mds <- cmdscale(diss[medoids, medoids], k=1)
      mdsk <- as.integer(factor(mdsk, levels=levels(mdsk)[order(mds)]))
    }
    kun <- length(unique(mdsk))
    if(kun!=k){
      warning(" [>] k value was adjusted to ", kun)
      k <- kun
      mdsk <- as.integer(factor(mdsk, levels=sort(unique(mdsk))))
    }
    medoids <- vector(mode="integer",length=k)
    #sortmds.seqdata$mdsk<-c(rep(1:m, each=r+1),rep({m+1}:k, each=r))
    ##pmdse <- 1:k
    #pmdse20<-1:20

    ##for each k
    for(i in 1:k){
      ##Which individuals are in group k
      ind <- which(mdsk==i)
      if(length(ind)==1){
        kmedoid.dist[ind] <- 0
        ##Index of the medoid sequence for each seq
        kmedoid.index[ind] <- ind
        medoids[i] <- ind
      }else{
        dd <- diss[ind, ind]
        ##Identify medoid
        kmed <- disscenter(dd, medoids.index="first", squared=squared)
        ##Distance to medoid for each seq
        kmedoid.dist[ind] <- dd[, kmed]
        ##Index of the medoid sequence for each seq
        kmedoid.index[ind] <- ind[kmed]
        medoids[i] <- ind[kmed]
      }
      index.list[[i]] <- ind
      dist.list[[i]] <- kmedoid.dist[ind]
      weights.list[[i]] <- rep(1,length(ind))
      ##Distance matrix for this group

    }
    #calculate R2
    R2 <-1-sum(kmedoid.dist^pow)/sum.gmedoid.dist
    med.names <- rownames(diss)[medoids]

    heights <- xtabs(~mdsk)/nrow(diss)
    at <- (cumsum(heights)-heights/2)/sum(heights)*length(heights)

  }

  #calculate F
  ESD <-R2/(k-1) # averaged explained variance
  #USD <-(1-R2)/(nrow(seqdata)-k) # averaged explained variance
  USD <-(1-R2)/(wsum-k) # averaged explained variance
  Fstat <- ESD/USD
  pvalue <- 1 - pf(Fstat, df1=k-1, df2=wsum-k)

  message(" [>] Pseudo/medoid-based-R2: ", format(R2))
  message(" [>] Pseudo/medoid-based-F statistic: ", format(Fstat),", p-value: ", format(pvalue) )
  ###   return(invisible(kmedoid.index))

  if (grp.meth=='prop'){
    retlist <- list(
      medoids = wg[,5],
      med.names = med.names,
      wg = wg,
      dist.list = dist.list,
      index.list = index.list,
      weights.list = weights.list,
      heights = rep(1/k,k), ## useless since they are all equal
      at = 1:k - .5,
      R2 = R2,
      Fstat = Fstat,
      pvalue = pvalue,
      sizes = c(ncase = ncase, wsum = wsum, k=k, gsize=gsize),
      grp.meth = grp.meth
    )
    class(retlist) <- c("dissrf","dissrfprop",class(retlist))
  } else {
    retlist <- list(
      medoids = medoids,
      med.names = med.names,
      dist.list = dist.list,
      index.list = index.list,
      weights.list = weights.list,
      kmedoid.index = kmedoid.index,
      kmedoid.dist = kmedoid.dist,
      ##seqtoplot = seqtoplot,
      mdsk = mdsk,
      heights = heights,
      #diss = diss,
      at = at,
      R2 = R2,
      Fstat = Fstat,
      pvalue = pvalue,
      sizes = c(ncase = ncase, k= k, ng=ng, r=r),
      grp.meth = grp.meth
    )
    class(retlist) <- c("dissrf","dissrfcrisp",class(retlist))
  }

  return(retlist)
}


#### methods

print.dissrf <- function(x, ...){
  #limit <- max(seqlength(seqdss(seqrf[["seqtoplot"]])))
  medoids <- x[["medoids"]]
  stat <- c(R2 = x[["R2"]],Fstat=x[["Fstat"]],pvalue=x[["pvalue"]])
  print(list(medoids = medoids,
              stat=stat,
              sizes = x[["sizes"]]),
              ...
  )

}



summary.dissrf <- function(object, dist.idx=1:10, ...){
  #limit <- max(seqlength(seqdss(seqrf[["seqtoplot"]])))
  medoids <- object[["medoids"]]
  dlist <- object[["dist.list"]]
  dweights <- object[["weights.list"]]
  g <- length(dlist)
  if (inherits(object,"dissrfprop")){
    dist.stat <- matrix(rep(NA,5*g), nrow=5)
    for (i in 1:g){
      dist.stat[,i] <- wtd.fivenum.tmr(dlist[[i]],dweights[[i]])
    }
  }
  else {
    dist.stat  <- sapply(dlist, fivenum)
  }
  k <- length(dlist)
  dmean <- stdev <- vector("double",length=k)

  for (i in 1:k) {
    dmean[i] <- wtd.mean(dlist[[i]], weights=dweights[[i]])
    if (length(dlist[[i]])>1)
      stdev[i] <- sqrt(wtd.var(dlist[[i]], weights=dweights[[i]]))
    else
      stdev[i] <- 0
  }
  dist.stat <- rbind(dist.stat,dmean,stdev)
  rownames(dist.stat) <- c("minimum","lower-hinge","median","upper-hinge","maximum","mean","stdev")
  if (min(dist.idx) < 1)
    dist.idx <- NULL
  else if (max(dist.idx) <= ncol(dist.stat))
    dist.stat <- dist.stat[,dist.idx]
  stat <- c(R2 = object[["R2"]],Fstat=object[["Fstat"]],pvalue=object[["pvalue"]])
  return(list(medoids = medoids,
              dist.stat = dist.stat,
              stat=stat,
              sizes = object[["sizes"]],
              grp.meth = object[["grp.meth"]]
  ))

}
