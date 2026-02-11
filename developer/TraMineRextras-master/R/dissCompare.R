## function for comparing sets of data by computing LRT and tail probability
##  from the matrix of distances between two groups
## version 1.0, December 2023

# diss  matrix of observed dissimilarities 
# group  dichotomous grouping variable 
# set   variable defining sets of row/columns of the diss matrix
# weights     vector of case weights

dissCompare <- function(diss,
  group, set=NULL,
  s=100, seed=36963, 
  stat="all", 
  squared="LRTonly",
  weighted=TRUE, 
  weights=NULL,
  #opt=NULL,
  BFopt=NULL
  )
{
  #require("gtools")
  #require("TraMineR")

  gc(FALSE)
  ptime.begin <- proc.time()

  diss <- as.matrix(diss)
    
  if (!is.null(set) & is.null(group)){
    stop("'set' not NULL while 'group' is NULL!")
  }
  if (length(group) != nrow(diss))
    stop("group length not equal to nrow(diss)!")
  
  w <- weights
  if (is.null(w))  
    w <- rep(1,nrow(diss))

  if (!is.logical(weighted)) {
    if (weighted != 'by.group')
      stop("weighted must be logical or 'by.group'")
    weight.by <- weighted
    weighted <- TRUE
  }
  else {
    weight.by <- 'global'
    if (!weighted) 
        w <- rep(1,nrow(diss))
  }
  if (length(w) != nrow(diss)){
    stop("length(w) must be equal to nrow(diss)!")
  }

  if (is.logical(squared))
    LRTpow <- 1
  else {
    if (squared != "LRTonly") stop("squared must be logical or 'LRTonly'")
    LRTpow <- 2
    squared <- FALSE
  }

  #is1.stslist <- inherits(seqdata,"stslist")
  #is2.stslist <- inherits(seqdata2,"stslist")

##  if (!is.list(ldiss)) {
##    ldiss <- list(ldiss)
##  }
##  if (!is.list(lgroup)) {
##    lgroup <- list(lgroup)
##  }
##  if (!is.list(lw))
##    lw <- list(lw)

  if (any(!stat %in% c("LRT","BIC","all")))
    stop("Bad stat value, must be one of 'LRT', 'BIC', or 'all'")

  if (any(stat=="all")) {
    is.LRT <- is.BIC <- TRUE
  }
  else{
    is.LRT <- "LRT" %in% stat
    is.BIC <- "BIC" %in% stat
  }

  gvar <- as.vector(group)
  if (is.null(set)) set <- rep(1,nrow(diss))
  #if (!is.null(set)){
    setvar <- as.vector(set)
    inotna <- which(!is.na(gvar) & !is.na(setvar))
    setvar <- setvar[inotna]
    setvar <- factor(setvar)
    lev.set <- levels(setvar)
  #}
  #else {
  #  inotna <- which(!is.na(gvar))
  #}
  ########
  ina <- nrow(diss) - length(inotna)
  if(ina > 0)
    message("[!!] ", ina, " cases removed because of NA values of the grouping variable(s)\n")
  ##########
  gvar <- gvar[inotna]
  gvar <- factor(gvar)
  lev.g <- levels(gvar)
  if (length(lev.g) == 1)
    stop("There is only one group among valid cases!")
  if (length(lev.g) > 2)
    stop("Currently dissCompare supports only 2 groups!")
  

  # prepare samples
  G = length(lev.set)  ## number of sets
  n = matrix(NA,nrow=G,ncol=2)
  # idx.a indexes of group a by set
  # idx.a indexes of group a by set
  idx.a = idx.b <- list()
  for (i in 1:G) {
    idx.a[[i]] <- which(group[set==lev.set[[i]]]==lev.g[1])
    idx.b[[i]] <- which(group[set==lev.set[[i]]]==lev.g[2])
  }

  for (i in 1:G) {
    if (length(idx.a[[i]])>=length(idx.b[[i]])) {
      n[i,1] <- length(idx.a[[i]])
      n[i,2] <- length(idx.b[[i]])
    }
    else {
      n[i,1] <- length(idx.b[[i]])
      n[i,2] <- length(idx.a[[i]])
    }
  }
  
  #print(idx.a)
  #print(idx.b)
  #print(n)

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
  #oopt <- opt
  multsple <- FALSE

  ## permute function from gtools that has become orphaned
  permute <- function(x) sample(x, size = length(x), replace = FALSE)

  ## Constructing vector of indexes of sampled cases
  #r.s1=r.s2 = list(rep(NA,G))
  for (i in 1:G) {
      if (n.n[i] > 0) {
        diss.i <- diss[c(idx.a[[i]],idx.b[[i]]),c(idx.a[[i]],idx.b[[i]])]
        #print(diss.i)
        weights <- w[c(idx.a[[i]],idx.b[[i]])]
        if (s==0) { # no sampling
          r1 <- 1:length(idx.a[[i]])
          r2 <- 1:length(idx.b[[i]]) + length(idx.a[[i]])
          #suppressMessages(diss <- seqdist(rbind(seq.a[[i]],seq.b[[i]]), method=method, weighted=weighted, ...))
          #weights <- c(attr(seq.a[[i]],"weights"),attr(seq.b[[i]],"weights"))
        suppressMessages(
            Results[i,] <-
              seqxcomp(r1, r2, diss.i, weights, is.LRT=is.LRT, is.BIC=is.BIC,
                 squared=squared, weighted=weighted, weight.by=weight.by,
                 LRTpow=LRTpow))
        }
        else { # sampling
          set.seed(seed)
          r.s1 <- c(permute(rep(1:m.n[i],ff.n1[i])),sample(1:m.n[i],r.n1[i],F))
          r.s2 <- c(permute(rep(1:n.n[i],k.n[i])),sample(1:n.n[i],r.n2[i],F))
          r.s1 = matrix(r.s1,ncol=s)
          r.s2 = matrix(r.s2,ncol=s)
    
          #if (is.null(oopt))
          #  opt <- ifelse(nrow(idx.a[[i]]) + nrow(idx.b[[i]]) > 2*s, 1, 2)
          ##message('opt = ',opt)
          #if (opt==2) {
          #  suppressMessages(diss <- seqdist(rbind(seq.a[[i]],seq.b[[i]]), method=method, weighted=weighted, ...))
          #  weights <- c(attr(seq.a[[i]],"weights"),attr(seq.b[[i]],"weights"))
          #}
    
          multsple <- nrow(r.s1) > 1 || multsple
        ### new complete samples without replacement of length s over G comparisons
          t<-matrix(NA,nrow=nrow(r.s1),ncol=nc)
          for (j in 1:nrow(r.s1)) {
    
            #if (opt==2) {
              r1 <- r.s1[j,]
              r2 <- r.s2[j,] + length(idx.a[[i]])
            #}
            suppressMessages(t[j,] <-
                seqxcomp(r1, r2, diss.i, weights, is.LRT=is.LRT, is.BIC=is.BIC,
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

