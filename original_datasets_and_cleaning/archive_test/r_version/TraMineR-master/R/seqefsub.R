## ========================================
## Find frequent subsequences
## ========================================

seqefsub <- function(eseq, str.subseq = NULL, min.support = NULL,
  pmin.support = NULL, constraint = seqeconstraint(), max.k = -1,
  weighted = TRUE, seq, strsubseq, minSupport, pMinSupport, maxK) {

  TraMineR.check.depr.args(alist(eseq = seq, str.subseq = strsubseq, min.support = minSupport,
    pmin.support = pMinSupport, max.k = maxK))

  if (!weighted) {
    ww <- seqeweight(eseq)
    totseq <- length(eseq)
    seqeweight(eseq) <- as.double(rep(1, totseq))
  } else {
    totseq <- sum(seqeweight(eseq))
  }
  ## message("Event sequence analysis module is still experimental")
  if (!is.seqelist(eseq)) {
    stop(" [!] eseq should be a seqelist. See help on seqecreate.")
  }
  if ((constraint$count.method==3)&(constraint$window.size==-1)) {
    stop(" [!] count method 3 or 'CMINWIN' requires a window.size.")
  }
  if (constraint$count.method==6) {
    warning(" [!] count method 6 is only for internal use.")
  }
  if (!inherits(constraint,"seqeconstraint")) {
    constraint=seqeconstraint()
    warning("[!] The constraint argument should be set using the seqeconstraint function. The provided constraint argument is ignored.")
    }
  # we are not looking for frequent but specific subsequences
  if (!is.null(str.subseq)) {
    subseq <- seqecreatesub(str.subseq, eseq)
    ret <- createsubseqelist(eseq, constraint, subseq, data.frame(),
                             type="user")
    constraint2 <- constraint
    constraint2$count.method <- 1

    ret2 <- createsubseqelist(eseq, constraint2, subseq, data.frame(),
                              type="user")
    ww <- seqeweight(eseq)
    ## Taking weights into account
    support <- colSums(ww*seqeapplysub(subseq=ret,
                                       constraint=constraint))
    support2 <- colSums(ww*seqeapplysub(subseq=ret2,
                                        constraint=constraint2))
    # namespace is needed to avoid name conflict with the argument seq
    ord <- base::seq(from=1,to=length(support))
    ## ord <- order(unlist(support), decreasing=TRUE)
    ret <- createsubseqelist(eseq,constraint,
                             subseq[ord],
                             data.frame(Support=(support2[ord]/totseq),
                                        Count=support[ord]),type="user")
    return(ret)
  } else if(is.null(min.support)) {
    if(is.null(pmin.support)) {
      stop(" [!] You should specify a minimum support through min.support or pmin.support argument")
    }
    min.support<-pmin.support*sum(seqeweight(eseq))
  }
  classname <- c("eseq")
  subseq <- .Call(C_tmrfindsubsequences,
                  unlist(list(eseq)),
                  as.double(c(constraint$max.gap)),
                  as.double(c(constraint$window.size)),
                  as.double(c(constraint$age.min)),
                  as.double(c(constraint$age.max)),
                  as.double(c(constraint$age.max.end)),
                  as.double(c(constraint$count.method)),
                  as.double(c(min.support)),
                  as.integer(c(max.k)),
                  classname)

  ord <- order(unlist(subseq[2]),decreasing=TRUE)
  support <- unlist(subseq[1])[ord]
  count <- unlist(subseq[2])[ord]
  ret <- createsubseqelist(eseq,
                           constraint,
                           unlist(subseq[3])[ord],
                           data.frame(Support=(support/totseq),
                                      Count=count))
  if (!weighted)
    {
      seqeweight(eseq) <- ww
    }
  return(ret)
}
