### function for linked polyadic sequence data
### Version 1.0.2 (29.12.20), Tim Liao, University of Illinois
### and Gilbert Ritschard


seqpolyads <- function (seqlist, a=1, method="HAM", ..., w=rep(1,ncol(combn(1:length(seqlist),2))),
              s=36963, T=1000, core=1, replace=TRUE, weighted=TRUE,
              with.missing=FALSE, rand.weight.type=1, role.weights=NULL, show.time=FALSE) {

  gc(FALSE)
  if (show.time){
    syst <- system.time(res <- seqpolyads.private(seqlist=seqlist, a=a, method=method, ..., w=w,
              s=s, T=T, core=core, replace=replace, weighted=weighted,
              with.missing=with.missing, rand.weight.type=rand.weight.type,
              role.weights=role.weights))
    print(syst)
  }
  else {
    res <- seqpolyads.private(seqlist=seqlist, a=a, method=method, ..., w=w,
              s=s, T=T, core=core, replace=replace, weighted=weighted,
              with.missing=with.missing, rand.weight.type=rand.weight.type,
              role.weights=role.weights)
  }
  return(res)

}


seqpolyads.private <- function (seqlist, a=1, method="HAM", ..., w=rep(1,ncol(combn(1:length(seqlist),2))),
              s=36963, T=1000, core=1, replace=TRUE, weighted=TRUE,
              with.missing=FALSE, rand.weight.type=1, role.weights=NULL, show.time=FALSE) {

  #gc(FALSE)
  #if (show.time) ptime.begin <- proc.time()

  if (!inherits(seqlist, "list") || length(seqlist)<2)
    TraMineR:::msg.stop("seqlist must be a list of at least two stslist objects")

  P <- length(seqlist)
  for (p in 1:P) {
    if (!inherits(seqlist[[p]],"stslist"))
      TraMineR:::msg.stop("At least one element of seqlist is not a stslist object")
  }

  if (is.null(role.weights)) role.weights <- 1/P
  else {
    if (length(role.weights) != P)
      TraMineR:::msg.stop("length of role.weights not equal to length of seqlist!")
  }

  n = nrow(seqlist[[1]])
  s = ncol(seqlist[[1]])
  alph = alphabet(seqlist[[1]], with.missing=with.missing)
  for (p in 2:P) {
    is.samealph <- identical(alphabet(seqlist[[p]], with.missing=with.missing), alph)
    if (nrow(seqlist[[p]]) != n | ncol(seqlist[[p]]) != s | !is.samealph)
      TraMineR:::msg.stop("All stslist objects must have same size and alphabet")
  }

  #require(doParallel)
  #require(TraMineR)
  #cl <- makeCluster(core, type="SOCK")
  if (core>1) {
    i<-0
    cl <- makePSOCKcluster(core)
    registerDoParallel(cl)
  }

  set.seed(s)
  c <- combn(1:P,2)
  d = ncol(c)
  #p.seq=
  p.temp=p1.temp <- list()
  polyads.dist=polyads.dummy=test.p <- array(0,c(1,n))

  seqall <- do.call("rbind", seqlist)
  ## sequence weights
  ## seqall has non null weights iff each stslist in seqlist has non null weights
  weights <- if(weighted) attr(seqall,"weights") else NULL
  if (is.null(weights)) weights <- rep(1,nrow(seqall))

  alldist <- suppressMessages(seqdist(seqall, method=method, with.missing=with.missing, weighted=weighted, ...))

  cj=0
  for (p in 2:P) cj[p] <- n * (p-1)

  ## Matrix of indexes of sampled sequences per generation
  if (core==1) {
    #l.m <- t(sapply(1:T, function(x) {cj + sample.int(n, size=P, replace=TRUE,)}))
    ## sapply is not faster than for
    l.m <- matrix(NA,T,P)
    for (i in 1:T) {
        l.m[i,] <- cj + sample.int(n, size=P, replace=TRUE,  prob=weights[1:n])
    }
  } else {
    l.m <- foreach (i=1:T, .combine='rbind') %dopar% {
      t(cj + sample.int(n, size=P, replace=TRUE, prob=weights[1:n]))
    }
  }

  if (a==1){
    if (core==1) {
      random.dist <- sapply(1:T, function(x) {sum(alldist[l.m[x,],l.m[x,]][upper.tri(matrix(NA,P,P))]*w)/d})
    } else {
      random.dist <- foreach (i=1:T, .combine='c') %dopar% {
        sum(alldist[l.m[i,],l.m[i,]][upper.tri(matrix(NA,P,P))]*w)/d
      }
    }
  } else if (a==2) {
    ## resample states in sampled sequences
    seqrandall <- as.matrix(seqall[as.vector(l.m),])
    s <- ncol(seqrandall)
    if (core==1){
      seqstrand <- t(sapply(1:(T*P), function(x) {sample(seqrandall[x,], size=s, replace=replace)}))
      ##seqstrand <- matrix(NA,nrow=T*P,ncol=s)
      ##for (i in 1:(T*P)) {
      ##  seqstrand[i,] <- t(sample(seqrandall[i,], size=s, replace=replace))
      ##}
    } else {
      seqstrand <- foreach (i=1:(T*P), .combine="rbind") %dopar% {
          t(sample(seqrandall[i,], size=s, replace=replace))
      }
    }

    #print(head(seqrandall))
    #print(head(seqstrand))
    void <- attr(seqall,"void")
    nr <- attr(seqall,"nr")
    right <- if (any(seqall==void)) 'DEL' else NA
    if (!is.na(right)) TraMineR:::msg.warn("Found voids in input sequences: dropping all right missings in the resampled sequences!")
    seqstrand[seqstrand %in% c(nr,void)] <- NA
    #print(alphabet(seqall))
    suppressMessages(seqstrand <- seqdef(seqstrand, alphabet=alphabet(seqall), nr=nr, right=right, void=void))
    suppressMessages(allrdist <-seqdist(seqstrand, method=method, with.missing=with.missing, ...))

    if (core==1){
      random.dist <- sapply(1:T, function(x) {sum(allrdist[cj+x,cj+x][upper.tri(matrix(NA,P,P))]*w)/d})
    } else {
      random.dist <- foreach (i=1:T, .combine='c') %dopar% {
        sum(allrdist[cj+i,cj+i][upper.tri(matrix(NA,P,P))]*w)/d
      }
    }
    rm(allrdist,seqstrand,seqrandall)
  } else {stop("Bad 'a' value")}


  if (core>1) stopCluster(cl)

  l.weights <- array(1,T)
  if (weighted){
    p.weights <- if (rand.weight.type == 2) apply(l.m,2,function(x){sum(weights[x])}) else 1
    for (i in 1:T) {
      l.weights[i] <- sum(weights[l.m[i,]]*role.weights/p.weights)
    }
  }
  l.weights <- l.weights/sum(l.weights)

  for (j in 1:n) {
    polyads.dist[j] <- sum(alldist[cj+j,cj+j][upper.tri(matrix(NA,P,P))]*w)/d
    #test.p[j] <- sum(polyads.dist[j]<random.dist)/T
    test.p[j] <- sum((polyads.dist[j]<random.dist)*l.weights)
    if (test.p[j]>0.95) polyads.dummy[j] <- 1
    else polyads.dummy[j] <- 0
  }
  mean.rand.dist <- sum(random.dist*l.weights)
  mean.U <- mean.rand.dist - polyads.dist
  U.p <- 2*pt(mean.U/{sd(random.dist)/sqrt(T)},T-1,lower.tail=F)

  mean.obs.dist <- sum(polyads.dist*weights[1:n])/sum(weights[1:n])
  mean.dist <- c(mean.obs.dist,mean.rand.dist)
  names(mean.dist) <- c("Obs","Rand")

  #if (show.time) print(proc.time()-ptime.begin)

  list(mean.dist=mean.dist,U=mean.U,U.tp=U.p,V=test.p,
       V.95=polyads.dummy,observed.dist=polyads.dist,random.dist=random.dist)
}
