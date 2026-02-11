## function for comparing sets of sequences by computing LRT and tail probability
## Tim Liao, University of Illionis, and Anette Fasang, Humboldt University
## version 1.0, March 2020

seqBIC <- function(seqdata, seqdata2=NULL, group=NULL, set=NULL,
  s=100, seed=36963, squared="LRTonly", weighted=TRUE, opt=NULL,
  BFopt=NULL, method, ...)
{
  return(seqCompare(seqdata, seqdata2, group, set, s, seed,
         stat="BIC", squared, weighted, opt, BFopt, method, ...))
}

seqLRT <- function(seqdata, seqdata2=NULL, group=NULL, set=NULL,
  s=100, seed=36963, squared="LRTonly", weighted=TRUE, opt=NULL,
  BFopt=NULL, method, ...)
{
  return(seqCompare(seqdata, seqdata2, group, set, s, seed,
         stat="LRT", squared, weighted, opt, BFopt, method, ...))
}

seqCompare <- function(seqdata, seqdata2=NULL, group=NULL, set=NULL,
  s=100, seed=36963, stat="all", squared="LRTonly",
  weighted=TRUE, opt=NULL, BFopt=NULL, method, ...)
{
  #require("gtools")
  #require("TraMineR")

  gc(FALSE)
  ptime.begin <- proc.time()

  if (is.null(seqdata2) & is.null(group)){
    stop("'seqdata2' and 'group' cannot both be NULL!")
  }
  if (!is.null(set) & is.null(group)){
    stop("'set' not NULL while 'group' is NULL!")
  }

  if (!is.logical(weighted)) {
    if (weighted != 'by.group')
      stop("weighted must be logical or 'by.group'")
    weight.by <- weighted
    weighted <- TRUE
  }
  else {
    weight.by <- 'global'
  }

  if (is.logical(squared))
    LRTpow <- 1
  else {
    if (squared != "LRTonly") stop("squared must be logical or 'LRTonly'")
    LRTpow <- 2
    squared <- FALSE
  }

  is1.stslist <- inherits(seqdata,"stslist")
  is2.stslist <- inherits(seqdata2,"stslist")

  if (is.list(seqdata) & !is1.stslist) {
    if (is2.stslist | length(seqdata) != length(seqdata2))
      stop("When 'seqdata' is a list, seqdata2 must be a list of same length")
    else {
      l <- length(seqdata)
      i <- 1
      while (i <= l) {
        if(!inherits(seqdata[[i]], "stslist") | !inherits(seqdata2[[i]], "stslist"))
           stop("At least one element of the seqdata lists is not a state sequence object!")
        i=i+1
      }
    }
  }
  else if (!is1.stslist) {
    stop("If not a list, 'seqdata' must be a state sequence object (stslist) created with seqdef()")
  }
  else if (!is.null(seqdata2) & !is2.stslist) {
    stop("If not a list, 'seqdata2' must be a state sequence object (stslist) created with seqdef()")
  }

  if (any(!stat %in% c("LRT","BIC","all")))
    stop("Bad stat value, must be one of 'LRT', 'BIC', or 'all'")

  if (any(stat=="all")) {
    is.LRT <- is.BIC <- TRUE
  }
  else{
    is.LRT <- "LRT" %in% stat
    is.BIC <- "BIC" %in% stat
  }

  if (!is1.stslist){
    seq1 <- seqdata
    seq2 <- seqdata2
  }
  else if (is1.stslist & !is.null(seqdata2)) {
    seq1 <- list(seqdata)
    seq2 <- list(seqdata2)
  }
  else if (is1.stslist & is.null(seqdata2)) {
    ## suppress cases with NA group values
    gvar <- as.vector(group)
    if (!is.null(set)){
      setvar <- as.vector(set)
      inotna <- which(!is.na(gvar) & !is.na(setvar))
      setvar <- setvar[inotna]
      setvar <- factor(setvar)
      lev.set <- levels(setvar)
    }
    else {
      inotna <- which(!is.na(gvar))
    }
    ########
    ina <- nrow(seqdata) - length(inotna)
    if(ina > 0)
      message("[!!] ", ina, " sequences removed because of NA values of the grouping variable(s)\n")
    ##########
    gvar <- gvar[inotna]
    gvar <- factor(gvar)
    lev.g <- levels(gvar)
    if (length(lev.g) == 1)
      stop("There is only one group among valid cases!")
    if (length(lev.g) > 2)
      stop("Currently seqLRT supports only 2 groups!")
    seqdata <- seqdata[inotna,]
    seq1 <- list()
    seq2 <- list()
    if (is.null(set)){
      seq1[[1]] <- seqdata[gvar==lev.g[1],]
      seq2[[1]] <- seqdata[gvar==lev.g[2],]
    }
    else {
      for (i in 1:length(lev.set)){
        seq1[[i]] <- seqdata[gvar==lev.g[1] & setvar==lev.set[i],]
        seq2[[i]] <- seqdata[gvar==lev.g[2] & setvar==lev.set[i],]
      }
    }
  }

  # prepare samples
  G = length(seq1)
  n = matrix(NA,nrow=G,ncol=2)
  seq.a = seq.b <- list(rep(NA,G))

  for (i in 1:G) {
    if (nrow(seq1[[i]])>=nrow(seq2[[i]])) {
      n[i,1] <- nrow(seq1[[i]])
      n[i,2] <- nrow(seq2[[i]])
      seq.a[[i]] <- seq1[[i]]
      seq.b[[i]] <- seq2[[i]]
    }
    else {
      n[i,1] <- nrow(seq2[[i]])
      n[i,2] <- nrow(seq1[[i]])
      seq.a[[i]] <- seq2[[i]]
      seq.b[[i]] <- seq1[[i]]
    }
  }

  n.n = apply(n,1,min)
  if (s>0) { # for s=0 we do not need that
    m.n = apply(n,1,max)
    f.n1 <- floor(s/m.n)
    ff.n1 <- sapply(f.n1, g<-function(x){max(1,x)})
    #r.n1 = s-m.n%%s
    r.n1 = ifelse(s<m.n, s - m.n%%s, s - f.n1*m.n)
    #k.n = floor((m.n+r.n1)/n.n)
    #r.n2 = (m.n+r.n1)-k.n*n.n
    k.n = floor((ff.n1*m.n+r.n1)/n.n)
    k.n[is.na(k.n)] <- 0
    r.n2 = (ff.n1*m.n+r.n1)-k.n*n.n
    r.n2[is.na(r.n2)] <- 0

    ##GR we have an error when s > min(m.n), because then we get some r.n1 > m.n
    ##GR Has been fixed, so we should never reach the following stops
    if(any(m.n<r.n1)) {
      ii <- which(m.n<r.n1)
      stop("rest r.n1 values greater than max m.n for i= ", ii, " s= ", s)
    }
    if(any(n.n<r.n2)) {
      ii <- which(n.n<r.n2)
      stop("rest r.n2 values greater than min n.n for i= ", ii, " s= ", s)
    }
  }

  nc <- ifelse(is.LRT & is.BIC, 4, 2)
  Results <- matrix(NA,G,nc)
  oopt <- opt
  multsple <- FALSE

  ## permute function from gtools that has become orphaned
  permute <- function(x) sample(x, size = length(x), replace = FALSE)

  ## Constructing vector of indexes of sampled cases
  #r.s1=r.s2 = list(rep(NA,G))
  for (i in 1:G) {
    if (n.n[i] > 0) {
        if (s==0) { # no sampling
          r1 <- 1:nrow(seq.a[[i]])
          r2 <- 1:nrow(seq.b[[i]]) + nrow(seq.a[[i]])
          suppressMessages(diss <- seqdist(rbind(seq.a[[i]],seq.b[[i]]), method=method, weighted=weighted, ...))
          weights <- c(attr(seq.a[[i]],"weights"),attr(seq.b[[i]],"weights"))
          suppressMessages(
            Results[i,] <-
              seqxcomp(r1, r2, diss, weights, is.LRT=is.LRT, is.BIC=is.BIC,
                 squared=squared, weighted=weighted, weight.by=weight.by,
                 LRTpow=LRTpow))
        }
        else { # sampling
          set.seed(seed)
          r.s1 <- c(permute(rep(1:m.n[i],ff.n1[i])),sample(1:m.n[i],r.n1[i],F))
          r.s2 <- c(permute(rep(1:n.n[i],k.n[i])),sample(1:n.n[i],r.n2[i],F))
          r.s1 = matrix(r.s1,ncol=s)
          r.s2 = matrix(r.s2,ncol=s)

          if (is.null(oopt))
            opt <- ifelse(nrow(seq.a[[i]]) + nrow(seq.b[[i]]) > 2*s, 1, 2)
          #message('opt = ',opt)
          if (opt==2) {
            suppressMessages(diss <- seqdist(rbind(seq.a[[i]],seq.b[[i]]), method=method, weighted=weighted, ...))
            weights <- c(attr(seq.a[[i]],"weights"),attr(seq.b[[i]],"weights"))
          }

          multsple <- nrow(r.s1) > 1 || multsple
        ### new complete samples without replacement of length s over G comparisons
          t<-matrix(NA,nrow=nrow(r.s1),ncol=nc)
          for (j in 1:nrow(r.s1)) {

            if (opt==2) {
              r1 <- r.s1[j,]
              r2 <- r.s2[j,] + nrow(seq.a[[i]])
            }
            else {
              seqA<-seq.a[[i]][r.s1[j,],]
              seqB<-seq.b[[i]][r.s2[j,],]
              seqAB <- rbind(seqA, seqB)
              wA <- attr(seqA,"weights")
              wB <- attr(seqB,"weights")
              weights <- c(wA,wB)
              r1 <- 1:length(r.s1[j,])
              r2 <- length(r.s1[j,]) + 1:length(r.s2[j,])
              suppressMessages(diss <- seqdist(seqAB, method=method, weighted=weighted, ...))
            }
            suppressMessages(t[j,] <-
                seqxcomp(r1, r2, diss, weights, is.LRT=is.LRT, is.BIC=is.BIC,
                  squared=squared, weighted=weighted, weight.by=weight.by,
                  LRTpow=LRTpow))
          }
          Results[i,]<-apply(t,2,mean)
        }
    }
  }
  colnames <- NULL
  if (is.LRT) colnames <- c("LRT", "p-value")
  if (is.BIC) {
    if (is.null(BFopt) && multsple) {
      BF2 <- exp(Results[,nc-1]/2)
      Results <- cbind(Results, BF2)
      colnames <- c(colnames, "Delta BIC", "Bayes Factor (Avg)", "Bayes Factor (From Avg BIC)")
    }
    else if (BFopt==1 && multsple) {
      colnames <- c(colnames, "Delta BIC", "Bayes Factor (Avg)")
    }
    else if (BFopt==2 && multsple) {
      BF2 <- exp(Results[,nc-1]/2)
      Results[,nc] <- BF2
      colnames <- c(colnames, "Delta BIC", "Bayes Factor (From Avg BIC)")
    }
    else {
      colnames <- c(colnames, "Delta BIC", "Bayes Factor")
    }
  }
  colnames(Results) <- colnames
  if(!is.null(set)) rownames(Results) <- lev.set

  #### Display elaspsed time ####

  ptime.end <- proc.time()
  time.begin <- as.POSIXct(sum(ptime.begin[1:2]), origin = "1960-01-01")
  time.end <- as.POSIXct(sum(ptime.end[1:2]), origin = "1960-01-01")
  time.elapsed <- format(round(difftime(time.end, time.begin), 3))

  message("elapsed time:", time.elapsed)

  return(Results)
}

####################
seqxcomp <- function(r1, r2, diss, weights, is.LRT,is.BIC, squared, weighted, weight.by,
                    LRTpow)
{
  #print(length(r1))
  #print(length(r2))
  #print(dim(diss))
  #print(length(weights))

  # compute some basic statistics
  #n1 = nrow(S1)
  #n2 = nrow(S2)
  n1 <- length(r1)
  n2 <- length(r2)
  n0 <- n1+n2
  #n.0=log((n0*n0-n0)/2) #not used
  dist.S=dist.S1=dist.S2<-vector()

  weighted <- weighted && !any(is.null(weights))

  ## weights
  if (weighted) {
    w1 <- weights[r1]
    w2 <- weights[r2]
    w <- c(w1,w2)
    ## normalize weights to respect group sizes
    if (weight.by == 'by.group') {
      w1 <- n1/sum(w1) * w1
      w2 <- n2/sum(w2) * w2
      w <- c(w1,w2)
    }
    nw <- sum(w)
    nw1 <- sum(w1)
    nw2 <- sum(w2)
  }
  else { # no weight
    nw <- n0
    nw1 <- n1
    nw2 <- n2
    w <- rep(1,nw)
    w1 <- rep(1,nw1)
    w2 <- rep(1,nw2)
  }

  ## SS <- nw*dissvar(diss[c(r1,r2),c(r1,r2)], weights=w, squared=squared) #
  ## SS1 <- nw1*dissvar(diss[r1,r1], weights=w1, squared=squared) #
  ## SS2 <- nw2*dissvar(diss[r2,r2], weights=w2, squared=squared) #

  ## Using dissvar does not allow the 'LRTonly' solution

  dist.S <-disscenter(diss[c(r1,r2),c(r1,r2)], weights=w, squared=squared) # calculate S distance to center
  dist.S1<-disscenter(diss[r1,r1], weights=w1, squared=squared) # calculate S1 distance to center
  dist.S2<-disscenter(diss[r2,r2], weights=w2, squared=squared) # calculate S2 distance to center

  SS <- sum(w*dist.S^LRTpow)
  SS1 <- sum(w1*dist.S1^LRTpow)
  SS2 <- sum(w2*dist.S2^LRTpow)

  res <- NULL

  LRT <- n0*(log(SS/n0) - log((SS1+SS2)/n0))

  if (is.LRT) {
    p.LRT <- pchisq(LRT,1,lower.tail=FALSE)
    res <- cbind(LRT, p.LRT)
  }

  if (is.BIC) {

    # compute BICs and adjusted BICs
    BIC <- LRT - 1*log(n0)
    # compute Bayes factors
    BF <- exp(BIC/2)

    res <- cbind(res, BIC, BF)
  }
  return(res)
}
