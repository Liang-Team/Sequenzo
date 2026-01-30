## ========================================
## Count number of time a subsequence appear in a sequence
## ========================================

seqeapplysub<-function(subseq,method=NULL,
                       constraint=NULL,rules=FALSE)
  {
    ## message("Event sequence analysis module is still experimental")
    if (!is.subseqelist(subseq)) {
      stop("subseq should be a subseqelist. See help on seqefsub.")
    }
    if (is.null(constraint)) {
      constraint <- subseq$constraint
    }
    if (!is.null(method)) {
      if (method=="count") {
        constraint$count.method <- 2
      } else if (method=="presence") {
        constraint$count.method <- 1
      } else if(method=="age") {
        constraint$count.method <- 6
      } else if(method=="COBJ") {
        constraint$count.method <- 1
      } else if(method=="CDIST_O") {
        constraint$count.method <- 2
      } else if(method=="CWIN") {
        constraint$count.method <- 3
      } else if(method=="CMINWIN") {
        constraint$count.method <- 4
      } else if(method=="CDIST") {
        constraint$count.method <- 5
      } else if (method%in%1:5) {
        constraint$count.method <- method
      }
    }
    if (!inherits(constraint, "seqeconstraint")) {
      constraint = seqeconstraint()
      warning("[!] The constraint argument should be set using the seqeconstraint function. The provided constraint argument is ignored.")
    }
    if(!rules)
      {
        return(.Call(C_tmrmatrixsubseqinseq,
                     unlist(list(subseq$subseq)),
                     unlist(list(subseq$eseq)),
                     as.double(c(constraint$max.gap)),
                     as.double(c(constraint$window.size)),
                     as.double(c(constraint$age.min)),
                     as.double(c(constraint$age.max)),
                     as.double(c(constraint$age.max.end)),
                     as.double(c(constraint$count.method))))
      }
    else {
      return(.Call(C_tmrmatrixsubseqinseq,
                   unlist(list(subseq$subseq)),
                   unlist(list(subseq$subseq)),
                   as.double(c(constraint$max.gap)),
                   as.double(c(constraint$window.size)),
                   as.double(c(constraint$age.min)),
                   as.double(c(constraint$age.max)),
                   as.double(c(constraint$age.max.end)),
                   as.double(c(constraint$count.method))))
    }
  }
