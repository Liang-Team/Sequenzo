# Author for TraMineR 2: Pierre-Alexandre Fonta (2016-2017)
## Fixes by Gilbert Ritschard (2017-2020)

seqdist <- function(seqdata, method, refseq = NULL, norm = "none", indel = "auto",
  sm = NULL, with.missing = FALSE, full.matrix = TRUE,
  kweights = rep(1.0, ncol(seqdata)), tpow = 1.0, expcost = 0.5, context,
  link = "mean", h = 0.5, nu, transindel = "constant", otto,
  previous = FALSE, add.column = TRUE, breaks = NULL, step = 1, overlap = FALSE,
  weighted = TRUE, global.pdotj=NULL, prox = NULL, check.max.size=TRUE,
  opt.args = list()) {

  gc(FALSE)
  ptime.begin <- proc.time()
  tol <- .Machine$double.eps^0.5 # Precision

  #### Check arguments with deprecated values ####

  # method
  # TODO Deprecated: remove in future versions.
  deprecated.methods <- c("OMopt", "LCSopt")
  if (method %in% deprecated.methods) {
    msg.warn(method, "is deprecated")
    if (method == "OMopt") {
      method <- "OM"
      msg("'method' is set to \"OM\" which is equivalent")
    } else if (method == "LCSopt") {
      method <- "LCS"
      msg("'method' is set to \"LCS\" which is equivalent")
    }
  }

  # norm
  if (is.logical(norm)) {
    norm <- if (isTRUE(norm)) "auto" else "none"
    msg.warn("'norm' has a deprecated value, TRUE changed into \"auto\", FALSE into \"none\"")
  }

  #### Check for arguments that need to be defined ####

  # method
  if (missing(method))
    msg.stop.miss("method")

  #### Check argument types ####

  # seqdata
  if (!inherits(seqdata, "stslist"))
    msg.stop("'seqdata' must be a state sequence object created with seqdef()")

  nseqs <- nrow(seqdata)
  alphabet <- alphabet(seqdata)
  nstates <- length(alphabet)
  seqs.dlens <- unique(seqlength(seqdata)) ## should we account for with.missing value (FALSE by default)?
  seqdata.nr <- attr(seqdata, "nr")

  # method
  # Add here new method names
  om.methods <- c("OM", "OMloc", "OMslen", "OMspell", "OMstran")
  methods <- c(om.methods, "HAM", "DHD", "CHI2", "EUCLID", "LCS", "LCP",
    "RLCP", "NMS", "NMSMST", "SVRspell", "TWED")
  if (! method %in% methods)
    msg.stop.in("method", methods)

  # refseq
  # refseq.type: "none", "sequence", "most frequent", "index", "sets"
  if (!is.null(refseq)) {
    ## if list of two sets of indexes, we will compute pairwise distances between the two sets
    if (inherits(refseq, "list") & length(refseq) > 1) {
      #if (method %in% c("OMstran"))
      #    msg.stop("refseq as a list not supported for  'OMstran'")
      if (length(refseq) > 2)
          msg.warn("Only first two elements of the 'refseq' list are used!")
      for (i in 1:2) {
        if (!is.positive.integers(refseq[[i]]))
          msg.stop("When a list, 'refseq' must contain two sets of indexes, ie of positive integer values.")
        if (max(refseq[[i]]) > nseqs)
          msg.stop("Some indexes in 'refseq' out of range.")
      }
      refseq.type <- "sets"
    } else if (inherits(refseq, "stslist")) {
      if (nrow(refseq) != 1)
        msg.stop("'refseq' must contain a (single) sequence")
      if (!identical(alphabet(refseq), alphabet))
        msg.stop("'refseq' and 'seqdata' must have the same alphabet")
      refseq.nr <- attr(refseq, "nr")
      if (!identical(seqdata.nr, refseq.nr))
        msg.stop("'refseq' and 'seqdata' must have same 'nr' attribute for missing values")
      refseq.type <- "sequence"
    } else if (is.a.positive.integer(refseq)) {
      if (refseq > nseqs)
        msg.stop("'refseq' must be less than the number of sequences in 'seqdata'")
      refseq.type <- if (refseq == 0) "most frequent" else "index"
    } else {
      msg.stop.na("refseq")
    }
  } else {
    refseq.type <- "none"
  }

  # checking for empty sequences
  sdur <- seqdur(seqdata, with.missing=with.missing)
  emptyseq <- which(is.na(sdur[,1]))
  if (length(emptyseq) > 0){
    if (method == "OMloc")
      msg.stop.sempty("OMloc", emptyseq)
    else
      msg.warn.sempty(emptyseq)
  }

  # with.missing
  has.seqdata.missings <- any(seqdata == seqdata.nr)
  has.refseq.missings <- if (refseq.type == "sequence" && any(refseq == refseq.nr)) TRUE else FALSE
  if (isTRUE(with.missing) && !has.seqdata.missings && !has.refseq.missings) {
    with.missing <- FALSE
    msg.warn("seqdist: 'with.missing' set as FALSE as 'seqdata' has no non-void missing values")
  }
  if (!isTRUE(with.missing) && (has.seqdata.missings || has.refseq.missings))
    msg.stop("'with.missing' must be TRUE when 'seqdata' or 'refseq' contain missing values")

  if (isTRUE(with.missing)) {
    nstates <- nstates + 1
    msg("including missing values as an additional state")
  }
  msg(nseqs, "sequences with", nstates, "distinct states")

  # norm
  # "auto" must be the first element
  # Add here new normalization method names
  norms <- c("auto", "none", "maxlength", "gmean", "maxdist", "YujianBo")
  if (! norm %in% norms)
    msg.stop.in("norm", norms)

  # indel
  # indel.type: "number", "vector", "auto"
  # Must be after including missing values as an additional state (nstates)
  if (is.a.number(indel)) {
    indel.type <- "number"
  } else if (is.vector(indel, mode = "numeric")) {
    if (length(indel) != nstates)
      msg.stop("when a vector, 'indel' must contain a cost for each state")
    indel.type <- "vector"
  } else if (length(indel)==1 && indel=="auto"){
      indel.type <- "auto"
  } else {
    msg.stop.na("indel")
  }

  # sm
  # Must be after sanity checks on 'indel'
  # Add here new seqcost() method names
  sm.methods <- c("TRATE", "CONSTANT", "INDELS", "INDELSLOG")
  # sm.type: "none", "matrix", "array", "method"
  if (!is.null(sm)) {
    if (is.matrix(sm)) {
      sm.type <- "matrix"
    } else if (is.array(sm)) {
      sm.type <- "array"
    } else if (is.character(sm)) {
      if (! sm %in% sm.methods)
        msg.stop.in("sm", sm.methods)
      sm.type <- "method"
    } else {
      msg.stop.na("sm")
    }
  } else {
    sm.type <- "none"
  }

  # prox
  # prox.type: "none", "matrix"
  if (!is.null(prox)) {
    if (is.matrix(prox)) {
      prox.type <- "matrix"
    } else {
      msg.stop.na("prox")
    }
  } else {
    prox.type <- "none"
  }

  # link
  # Add here new link method names
  links <- c("none", "mean", "gmean")
  if (! link %in% links)
    msg.stop.in("link", links)

  # step
  if (!is.a.positive.integer(step))
    msg.stop("'step' must be a positive integer")

  #### Check arguments not yet implemented ####

  # method
  #if (sm.type == "method" && sm %in% c("INDELS", "INDELSLOG") && method == "DHD")
  ##  msg.stop.impl("sm", method, values = c("INDELS", "INDELSLOG")) # See seqcost()

  # refseq
  #if (refseq.type != "none" && method %in% c("OMstran", "CHI2", "EUCLID"))
  #if (refseq.type != "none" && method %in% c("CHI2", "EUCLID"))
  #  msg.stop.impl("refseq", method)
  #if (refseq.type == "sequence" && ! method %in% c("OM", "OMstran", "HAM", "DHD", "LCS", "LCP", "RLCP", "CHI2", "EUCLID"))
  #  msg.stop.impl("refseq", method, when = "it is an external sequence object")

  # norm: all but  SVRspell, NMS, NMSMST
  if (norm != "none" && ! method %in% c("OM", "OMloc", "OMstran", "OMspell", "OMslen", "TWED", "HAM", "DHD", "CHI2", "EUCLID", "LCS", "LCP", "RLCP"))
  ##if (norm != "none" && ! method %in% c("OM", "HAM", "DHD", "LCS", "LCP", "RLCP"))
    msg.stop.impl("norm", method)

  #### Check method specific arguments ####

  # OMloc, OMspell
  if (method %in% c("OMloc", "OMspell") && expcost < 0)
    msg.stop("'expcost' must be positive")

  # OMloc
  if (method == "OMloc") {
    if (missing(context)) {
      context <- 1 - 2 * expcost # Does not work in the function declaration
      msg("context set to 1 - 2 * expcost =", context)
    }
    if (context < 0)
      msg.stop("'context' must be positive ('expcost' must be in [0, 0.5])")
    msg("2 * expcost + context =", 2 * expcost + context)
  }
  # OMslen
  else if (method == "OMslen") {
    ##if (isTRUE(with.missing))  ## GR Feb 2020 Now works with missings
    ##  msg.stop("'with.missing' is not supported for OMslen")
    if (link == "none")
      msg.stop.miss("link")
    if (! link %in% c("mean", "gmean"))
      msg.stop.na("link")
    # According to Marteau, we should have h >= 0
    if (!is.a.number(h) || h < 0)
      msg.stop("'h' must be a number greater than or equal to 0")
  }
  # OMstran
  else if (method == "OMstran") {
    ##if (isTRUE(with.missing))
    ##  msg.stop("'with.missing' is not supported for OMstran")
    if (missing(otto))
      msg.stop.miss("otto")
    else if (!is.a.number(otto) || otto < 0 || otto > 1)
      msg.stop("'otto' must be a number in ]0, 1]")
    # TODO Implement in future versions
    ##if (length(seqs.dlens) > 1)
    ##  msg.stop(method, "currently works only with sequences of equal length")
  }
  # DHD
  else if (method == "DHD") {
    if (sm.type == "method" && sm == "CONSTANT")
      msg.stop("'sm = \"CONSTANT\"' is not relevant for DHD, consider HAM instead")
  }
  # CHI2 + EUCLID
  else if (method %in% c("CHI2", "EUCLID")) {
    if (!is.null(breaks)) {
      msg.warn.ign2("step", "breaks")
      msg.warn.ign2("overlap", "breaks")
    } else if (isTRUE(overlap) && step %% 2 != 0) {
      msg.stop("'step' must be even when 'overlap = TRUE'")
    }
  }
  # NMS + NMSMST + SVRspell
  else if (method %in% c("NMS", "NMSMST", "SVRspell")) {
    if (!is.vector(kweights, mode = "numeric") || any(kweights < 0))
      msg.stop("'kweights' must be a vector of positive numbers")
    if (length(kweights) == 1){
      kweights <- rep(kweights, ncol(seqdata))
      msg.warn("scalar kweights transformed into vector rep(kweights, ncol(seqdata))")
    }
    if (length(kweights) < ncol(seqdata))
      msg.warn("length(kweights) < ncol(seqdata), longer subsequences will be ignored!")
  }
  # TWED
  else if (method == "TWED") {
    if (missing(nu))
      msg.stop.miss("nu")
    # According to Marteau, we should have h >= 0 and nu > 0
    if (!is.a.number(h) || h < 0)
      msg.stop("'h' must be a number greater than or equal to 0")
    if (!is.a.number(nu) || nu <= 0)
      msg.stop("'nu' must be a number strictly greater than 0")
  }

  # HAM, DHD
  if (method %in% c("HAM", "DHD")) {
    if (length(seqs.dlens) > 1)
      msg.stop(method, "is not defined for sequences of different length")
  }

  # NMS, SVRspell
  if (method %in% c("NMS", "SVRspell") && sm.type != "none")
    msg.stop("use 'prox' instead of 'sm'")

  # OMloc, OMslen, OMspell, HAM, DHD, CHI2, EUCLID, LCS, LCP, RLCP, NMS, NMSMST, SVRspell, TWED
  ##if (! method %in% c("OM", "OMstran") && indel.type == "vector")
  ##if (method %in% c("OMslen","OMspell", "TWED") && indel.type == "vector"){

  # TWED
  if (method == "TWED" && indel.type == "vector"){
    msg.warn("indel vector not supported by the chosen method, max(indel) used instead!")
    indel <- max(indel)
    indel.type <- "number"
  }


  #### Configure norm ####

  # OMslen
  #if (method == "OMslen" && ! norm %in% c("none", "auto", "maxdist", "YujianBo"))
  #  msg.stop("For",method,"norm can only be one of 'none', 'auto', 'maxdist', or 'YujianBo'")

  if (method %in% c("EUCLID","CHI2") && ! norm %in% c("auto", "none"))
    msg.stop("For",method,"norm can only be one of 'none' or 'auto'")

  if (norm == "auto") {
    if (method %in% c("OM", "HAM", "DHD"))
      norm <- "maxlength"
    else if (method %in% c("LCS", "LCP", "RLCP"))
      norm <- "gmean"
    else if (method %in% c("OMloc", "OMstran", "OMspell", "OMslen", "TWED"))
      norm <- "YujianBo"
    else if (! method %in% c("CHI2", "EUCLID"))
      msg.stop.ie("no known normalization method to select automatically for", method)
  }

  # Must be after checking the valid values of norm for CHI2 and EUCLID
  if (method %in% c("CHI2", "EUCLID"))
    norm.chi2euclid <- switch(norm, auto = TRUE, none = FALSE)

  #### Configure prox ####

  # NMS, SVRspell
  if (method %in% c("NMS", "SVRspell")) {
    if (prox.type == "matrix") {
      if (nrow(prox) != nstates || ncol(prox) != nstates)
        msg.stop("'prox' must be of size", nstates, "x", nstates)
      eg <- eigen(prox)
      if (any(eg$values < -tol))
        msg.warn("'prox' is not positive semi-definite. Eigen values: ",
          paste(round(eg$values, 3), collapse = " "))
      rm(eg)
    } else if (prox.type == "none") {
      if (method == "SVRspell") {
        # Autogenerate prox (neutral)
        msg("creating a neutral 'prox' (identity matrix)")
        prox.type <- "matrix"
        prox <- diag(nstates)
      }
    } else {
      msg.stop.miss("prox")
    }
  }

  #### Configure sm and indel ####

  if (indel.type =="auto" && sm.type == "matrix"){
    if (method == "TWED")
      indel <- 2*max(sm) + nu + h
    else
      indel <- max(sm)/2
    indel.type <- "number"
  }

  # LCS
  if (method == "LCS") {
    # Autogenerate sm
    msg("creating a 'sm' with a substitution cost of 2")
    sm.type <- "matrix"
    sm <- seqsubm(seqdata, "CONSTANT", with.missing=with.missing, cval = 2, miss.cost = 2)
  }
  # OM, OMloc, OMslen, OMspell, OMstran, HAM, DHD, TWED
  else if (method %in% c(om.methods, "HAM", "DHD", "TWED")) {
    # matrix
    if (sm.type == "matrix") {
      if (method %in% c(om.methods, "TWED"))
        checkcost(sm, seqdata, with.missing, indel)
      else if (method == "HAM")
        checkcost(sm, seqdata, with.missing)
      else
        msg.stop.na("sm")
    }
    # array
    else if (sm.type == "array") {
      if (method == "DHD")
        checkcost(sm, seqdata, with.missing)
      else
        msg.stop.na("sm")
    }
    # method
    else if (sm.type == "method") {
      tv <- FALSE
      cost <- NULL
      if (sm %in% c("INDELS", "INDELSLOG")) {
      #  cost <- NULL
        if(method == "DHD") tv <- TRUE
      } else if (sm == "TRATE") {
        if (method == "OM") {
          cost <- 2
          #tv <- FALSE
        } else if (method == "HAM") {
          cost <- 2
          #tv <- FALSE
        } else if (method == "DHD") {
          cost <- 4
          tv <- TRUE
          #sm.type <- "array" # Not used. Should be here if it changes.
        } #  else {
          #  msg.stop.na("sm")
          #}
      } else if (sm == "CONSTANT") { ## method cannot be DHD, message issued above
        if (method == "HAM") {
          cost <- 1
          #tv <- FALSE
        } else {
          cost <- 2
          #tv <- FALSE
        } #else {
          # msg.stop.na("sm")
          #}
      } #else {
        #msg.stop.na("sm")
      #}
      msg("Computing sm with seqcost using ",sm)
      sm <- seqcost(seqdata, sm, with.missing = with.missing, cval = cost, miss.cost = cost, time.varying = tv, weighted = weighted)

      if (indel.type=="auto"){
        indel <- sm$indel
        indel.type <- ifelse (length(indel) > 1, "vector", "number")
        #if (method %in% c("OMslen", "OMspell", "TWED") && indel.type == "vector")
        if (method == "TWED" ){
          indel <- 2*max(sm$sm) + nu + h
          indel.type <- "number"
        }
        msg("generated an indel of type ",indel.type)
      }
      sm <- sm$sm
      rm(cost)
      rm(tv)
    }
    # none
    else {
      if (method == "HAM") {
        # Autogenerate sm
        msg("creating a 'sm' with a single substitution cost of 1")
        sm <- seqsubm(seqdata, "CONSTANT", with.missing=with.missing, cval = 1, miss.cost = 1)
      } else if (method == "DHD") {
        # Autogenerate sm
        msg("creating a 'sm' with the costs derived from the transition rates")
        #sm.type <- "array" # Not used. Should be here if it changes.
        sm <- seqsubm(seqdata, "TRATE", with.missing=with.missing, cval = 4, miss.cost = 4, time.varying = TRUE, weighted = weighted)
      } else {
        msg.stop.miss("sm")
      }
    }
  } # CHI2, EUCLID, LCP, RLCP, NMS, NMSMST, SVRspell do not use sm
  else if (! method %in% c("CHI2", "EUCLID", "LCP", "RLCP", "NMS", "NMSMST", "SVRspell")) {
    msg.stop.ie("no known 'sm' preparation for", method)
  }

  #### Pre-process data (part 1/2) ####

  # TODO Temporary fix because stringdist C++ code uses a sequence index, not a sequence object!
  if (refseq.type == "sequence") {
    seqs.lens.max <- max(seqs.dlens)
    refseq.len <- seqlength(refseq)[1, 1]
    ##refseq.mat <- as.matrix(refseq)
    if (refseq.len > seqs.lens.max)
      msg.stop("'refseq' cannot be longer than the longest 'seqdata' sequence!")
   # if (refseq.len < seqs.lens.max) {
   #   void <- attr(seqdata, "void")
      #refseq.mat.ext <- matrix(void, nrow = 1, ncol = seqs.lens.max)
      #for (i in 1:refseq.len)
      #  refseq.mat.ext[i] <- refseq.mat[i]
      #refseq.mat <- refseq.mat.ext
    #}
    # Tell seqdef() that the seqdata.nr/refseq.nr code is the one for missing values
    ##seqdata <- suppressMessages(seqdef(rbind(as.matrix(seqdata), refseq.mat),
    ##  alphabet=alphabet(seqdata),
    ##  missing = seqdata.nr))
    ## We use the rbind method available since v2.0-16
    ##  and set a zero weight for refseq
    if (is.null(attr(seqdata,"weights")) || !weighted) {
      attr(seqdata,"weights") <- rep(1,nrow(seqdata))
      weighted <- TRUE
    }
    attr(refseq,"weights") <- 0
    rownames.ori <- rownames(seqdata)
    seqdata <- rbind(seqdata,refseq)
    refname <- if ("ref" %in% rownames.ori) rownames(seqdata)[nrow(seqdata)] else "ref"
    rownames(seqdata) <- c(rownames.ori,refname)
  }

  # Transform the alphabet into numbers
  seqdata.num <- seqnum(seqdata, with.missing)
  # Keep only distinct sequences

  if (refseq.type == "sets") {
    dseqs.num1 <- unique(seqdata.num[refseq[[1]],])
    nunique1 <- nrow(dseqs.num1)
    dseqs.num2 <- unique(seqdata.num[refseq[[2]],])
    nunique2 <- nrow(dseqs.num2)
    dseqs.num <- rbind(dseqs.num1,dseqs.num2)
  } else {
    dseqs.num <- unique(seqdata.num)
  }
  # Check that dseqs.num does not exceed the max allowed number
  if (check.max.size){
    max.allowed.seq <- ifelse(refseq.type == "none",
                          floor(sqrt(.Machine$integer.max)),
                          .Machine$integer.max - 1)
    if (refseq.type == "sets") {
      if((sqrt(nunique1) * sqrt(nunique2)) > max.allowed.seq)
        msg.stop("number of ",nunique1, " and ", nunique2, " unique sequences too large for max allowed distances ", max.allowed.seq)
    } else if (nrow(dseqs.num) > max.allowed.seq) {
      msg.stop(nrow(dseqs.num), " unique sequences exceeds max allowed of ", max.allowed.seq)
    }
  }

  #### Handle reference sequence ####

  # Find the index of the corresponding representative (distinct) sequence
  # Note: Must be before dseqs.num modification for OMspell, NMSMST, SVRspell

  if (refseq.type == "sets") {
    seqdata.didxs1 <- match(seqconc(seqdata.num[refseq[[1]],]), seqconc(dseqs.num1))
    seqdata.didxs2 <- match(seqconc(seqdata.num[refseq[[2]],]), seqconc(dseqs.num2))
  } else {
    seqdata.didxs <- match(seqconc(seqdata.num), seqconc(dseqs.num))
  }

  if (refseq.type != "none") {
    # sets
    if (refseq.type == "sets") {
      refseq.raw <- refseq
      if (method %in% c("OMstran","CHI2", "EUCLID"))
        refseq.id <- refseq ## list of the two sets
      else
        refseq.id <- c(nunique1, nunique1 + nunique2) ## vector of sets sizes
    }
    # sequence
    else if (refseq.type == "sequence") {
      # TODO Temporary fix because stringdist C++ code uses a sequence index, not a sequence object!
      refseq.raw <- refseq
      if (method %in% c("OMstran","CHI2", "EUCLID"))
        refseq.id <- nseqs + 1
      else
        refseq.id <- seqdata.didxs[nseqs + 1]
    }
    # most frequent
    else if (refseq.type == "most frequent") {
      mfseq.freq <- seqtab(seqdata.num, idxs = 1)
      mfseq.idxs <- suppressMessages(seqfind(mfseq.freq, seqdata.num))
      msg("the most frequent sequence appears", length(mfseq.idxs), "time(s)")
      mfseq.idx <- mfseq.idxs[1]
      refseq.raw <- seqdata[mfseq.idx, ]
      if (method %in% c("OMstran","CHI2", "EUCLID"))
        refseq.id <- mfseq.idx
      else
        refseq.id <- seqdata.didxs[mfseq.idx]

      rm(mfseq.freq)
      rm(mfseq.idxs)
      rm(mfseq.idx)
    }
    # index
    else if (refseq.type == "index") {
      refseq.raw <- seqdata[refseq, ]
      refseq.id <- seqdata.didxs[refseq]
    }
    else {
      msg.stop.ie("no known preparation for this 'refseq' type")
    }
    if (refseq.type == "sets") {
      msg("pairwise measures between two subsets of sequences of sizes ",length(refseq[[1]])," and ",length(refseq[[2]]))
    } else {
      refseq.sps <- suppressMessages(seqformat(refseq.raw, from = "STS", to = "SPS", compress = TRUE))
      msg("using reference sequence", refseq.sps)
      rm(refseq.sps)
    }
    rm(refseq.raw)
  }

  #### Compute method specific values ####

  if (method %in% c("OMslen","OMspell") && indel.type == "number"){
    indel <- rep(indel, nstates)
    indel.type <- "vector"
  }


  # OMslen
  if (method == "OMslen") {
    dseqs.dur <- seqdur(dseqs.num, with.missing=with.missing)
    dur.mat <- matrix(0, nrow = nrow(dseqs.num), ncol = ncol(dseqs.num))
    for (i in 1:nrow(dseqs.num)) {
      y <- dseqs.dur[i, !is.na(dseqs.dur[i, ])]
      if(length(y) > 0) dur.mat[i, 1:sum(y)] <- rep(y, times = y)
    }
    dur.mat <- dur.mat ^ (-1 * h)
    rm(dseqs.dur)
    rm(y)
  }
  # OMspell, NMSMST (part 1/2), SVRspell (part 1/2)
  # Redefined dseqs.num
  else if (method %in% c("OMspell", "NMSMST", "SVRspell")) {
    dseqs.dur <- seqdur(seqdata, with.missing) ^ tpow # Do not use dseqs.num
    dseqs.oidxs <- match(seqconc(dseqs.num), seqconc(seqdata.num))
    c <- if (method == "OMspell") 1 else 0
    dseqs.dur <- dseqs.dur[dseqs.oidxs,,drop=FALSE] - c
    seqdata.dss <- seqdss(seqdata, with.missing)
    dseqs.num <- seqnum(seqdata.dss[dseqs.oidxs,,drop=FALSE], with.missing)
    if (method == "OMspell") {
      seqlength <- seqlength(seqdata, with.missing)
      seqlength <- seqlength[dseqs.oidxs]
    }
    rm(dseqs.oidxs)
    rm(c)
    rm(seqdata.dss)
  }
  # HAM, DHD
  else if (method %in% c("HAM", "DHD")) {
    if (method == "HAM")
      #sm.type <- "array" # Not used. Should be here if it changes.
      sm <- adaptSmForHAM(sm, nstates, ncol(seqdata))
    # Maximum possible cost of the Hamming distance
    max.cost <- 0
    for (i in 1:max(seqs.dlens))  ## actually seqs.dlens has here only one value
      max.cost <- max.cost + max(sm[, , i])
  }

  # NMS, NMSMST (part 2/2), SVRspell (part 2/2)
  # Modified dseqs.num for NMSMST and SVRspell
  if (method %in% c("NMS", "NMSMST", "SVRspell")) {
    ncols <- ncol(dseqs.num)
    nmin <- min(ncols, length(kweights))
    kweights2 <- double(ncols)
    kweights2[1:nmin] <- kweights[1:nmin]
    rm(ncols)
    rm(nmin)
  }


  rm(seqdata.num)

  #### Pre-process data (part 2/2) ####

  # Modified dseqs.num for OMspell, NMSMST, SVRspell
  ndn <- nrow(dseqs.num)
  # TODO Temporary fix because seqdist2 C++ code use a sequence index, not a sequence object!
  #ndn <- if (refseq.type == "sequence") dn-1 else dn
  incl.refseq <- if (refseq.type == "sequence") "(including refseq)" else ""
  seq.or.spell <- if (method %in% c("OMspell", "SVRspell")) "spell sequences" else "sequences"
  msg(ndn, "distinct ", seq.or.spell, incl.refseq)
  ##rm(dn)
  rm(ndn)
  rm(seq.or.spell)

  # Compute the sequence lengths
  # Modified dseqs.num for OMspell, NMSMST, SVRspell
  dseqs.lens <- seqlength(dseqs.num)  ## dseqs.lens  length of dss
  ds <- if (method %in% c("OMspell", "NMSMST", "SVRspell")) "spell " else ""
  # TODO Temporary fix because seqdist2 C++ code use a sequence index, not a sequence object!
  dl <- if (refseq.type == "sequence" & length(dseqs.lens)>1) dseqs.lens[-length(dseqs.lens)] else dseqs.lens
  msg0("min/max ", ds, "sequence lengths: ", min(dl), "/", max(dl))
  rm(ds)
  rm(dl)

  #### Configure params ####

  if (method=="OMspell" & is.list(opt.args)){
    if (!is.null(opt.args[["tokdep.coeff"]])){
        tokdep.coeff <- opt.args[["tokdep.coeff"]]
        if (length(tokdep.coeff) != length(indel))
            msg.stop("tokdep.coeff should be a vector of same length as indel")
        else method <- "OMtspell"
    }
  }



  params <- list()
  nstates <- as.integer(nstates)

  # OM
  if (method == "OM") {
    params[["alphasize"]] <- nstates
    params[["scost"]] <- sm

    if (indel.type == "number") {
      params[["indel"]] <- indel
    } else if (indel.type == "vector") {
      # Also executed when 'method = "OMstran"' (no matter the type of 'indel')
      params[["indels"]] <- indel
      params[["indelmethod"]] <- as.integer(0)
      params[["indel"]] <- max(indel) # GR for normalization. TODO Remove from C++ code. Not used. Avoid a NPE.
    } else {
      msg.stop.ie("no known configuration for this 'indel' type for OM")
    }
  }
  # OMloc
  else if (method == "OMloc") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- max(sm) * expcost + context  ## for normalization, indels computed in C++
    params[["indelmethod"]] <- as.integer(1)
    params[["scost"]] <- sm
    params[["localcost"]] <- context
    params[["timecost"]] <- expcost
  }
  # OMslen
  else if (method == "OMslen") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- max(indel)
    params[["indels"]] <- indel
    params[["seqdur"]] <- as.double(dur.mat)

    if (link == "mean") {
      params[["scost"]] <- sm / 2
      params[["sublink"]] <- as.integer(1)
    } else if (link == "gmean") {
      params[["scost"]] <- sm
      params[["sublink"]] <- as.integer(0)
    } else {
      msg.stop.ie("no known configuration for this 'link' value for OMslen")
    }

    rm(dur.mat)
  }
  # OMspell
  else if (method == "OMspell") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- max(indel)
    params[["indels"]] <- indel
    params[["scost"]] <- sm
    params[["seqdur"]] <- as.double(dseqs.dur)
    params[["timecost"]] <- expcost
    params[["seqlength"]] <- as.integer(seqlength)
    rm(dseqs.dur)
  }
  # OMtspell
  else if (method == "OMtspell") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- max(indel)
    params[["indels"]] <- indel
    params[["scost"]] <- sm
    params[["seqdur"]] <- as.double(dseqs.dur)
    params[["timecost"]] <- expcost
    params[["seqlength"]] <- as.integer(seqlength)
    params[["tokdepcoeff"]] <- tokdep.coeff
    rm(dseqs.dur)
  }
  # HAM + DHD
  else if (method %in% c("HAM", "DHD")) {
    params[["alphasize"]] <- nstates
    params[["maxdist"]] <- max.cost
    params[["scost"]] <- sm
    rm(max.cost)
  }
  # LCS
  else if (method == "LCS") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- 1
    params[["scost"]] <- sm
  }
  # LCP
  else if (method == "LCP") {
    params[["sign"]] <- as.integer(1)
  }
  # RLCP
  else if (method == "RLCP") {
    params[["sign"]] <- as.integer(-1)
  }
  # NMS + NMSMST + SVRspell
  else if (method %in% c("NMS", "NMSMST", "SVRspell")) {
    params[["distMethod"]] <- as.integer(2)
    params[["distTransform"]] <- as.integer(0) # TODO Remove from C++ code
    params[["kweights"]] <- as.double(kweights2)

    if (method != "NMS") {
      params[["seqdur"]] <- as.double(dseqs.dur)
      rm(dseqs.dur)
    }

    if (method != "NMSMST" && prox.type == "matrix") {
      params[["alphasize"]] <- nstates
      params[["softmatch"]] <- prox
    }
  }
  # TWED
  else if (method == "TWED") {
    params[["alphasize"]] <- nstates
    params[["indel"]] <- max(indel)
    params[["lambda"]] <- h
    params[["nu"]] <- nu
    params[["scost"]] <- sm
  }

  #### Configure method.num ####

  # TODO Assign a number after integration with C++ code for OMstran, CHI2, EUCLID
  method.num <-
    switch(method,
      OM = if (indel.type == "vector") 7 else 1, # TODO Align C++ logic with the theory
      OMloc = 7,
      OMslen = 10,
      OMspell = 8,
      #OMstran = ,
      HAM = 4,
      DHD = 4,
      #CHI2 = ,
      #EUCLID = ,
      LCS = 1,
      LCP = 2,
      RLCP = 2,
      NMS = if (prox.type == "matrix") 12 else 5,
      NMSMST = 6,
      SVRspell = 13,
      TWED = 14,
      OMtspell = 15
      )

  # Transform the sequence object into a matrix
  # Modified dseqs.num for OMspell, NMSMST, SVRspell
  dseqs.mat <- seqasnum(dseqs.num, with.missing)

  rm(dseqs.num)

  #### Compute distances ####

  nm <- if (norm != "none") paste("", norm,"normalized") else ""
  msg0("computing distances using the ", method, nm, " metric")
  rm(nm)

  # CHI2, EUCLID
  if (method %in% c("CHI2", "EUCLID")) {
    # TODO Integrate into C++ code instead of using CHI2()
    is.EUCLID <- if (method == "EUCLID") TRUE else FALSE
    if (refseq.type == "none") {
      distances <- CHI2(seqdata, breaks = breaks, step = step,
        with.missing = with.missing, norm = norm.chi2euclid,  weighted = weighted,
        overlap = overlap, euclid = is.EUCLID, global.pdotj=global.pdotj)
      if (full.matrix) {
        result <-  dist2matrix(distances)
        dimnames(result) <- list(rownames(seqdata),rownames(seqdata))
      } else {
        result <- distances
      }
    } else { ## dist to ref
      result <- CHI2(seqdata, breaks = breaks, step = step,
        with.missing = with.missing, norm = norm.chi2euclid,  weighted = weighted,
        overlap = overlap, euclid = is.EUCLID, global.pdotj=global.pdotj, refseq=refseq.id)
      ##names(result) <- rownames(seqdata)
      if (refseq.type == "sets") {
        dimnames(result) <- list(rownames(seqdata)[refseq[[1]]],rownames(seqdata)[refseq[[2]]])
      } else {
        names(result) <- rownames(seqdata)
        if (refseq.type == "sequence") result <- result[-length(result)]
      }
    }
  }
  # OMstran
  else if (method == "OMstran") {
    # TODO Integrate into C++ code instead of using OMstran()
    # OMstran() calls seqdist() with 'method = "OM"'

    if (length(seqstatl(seqdata))==1){  ## only identical constant sequences
                                        ## dist is 0 as with "HAM"
        refseq.arg <- if (refseq.type == "none") refseq else refseq.id
        result <- suppressMessages(seqdist(seqdata, method="HAM", refseq = refseq.arg))
        rm(refseq.arg)
        if(refseq.type=="sequence")
           result <- result[-length(result)]
    }
    else {
        # Dissimilarities with a reference sequence
        if (refseq.type != "none") {
          result <- OMstran(seqdata, indel = indel, sm = sm,
            full.matrix = full.matrix, transindel = transindel, otto = otto,
            previous = previous, add.column = add.column, with.missing=with.missing,
            weighted = weighted, refseq = refseq.id, norm = norm)

          #names(result) <- rownames(seqdata)

          # TODO Temporary fix because seqdist2 C++ code use a sequence index, not a sequence object!
          if (refseq.type == "sequence")
            result <- result[-length(result)]
        }
        else {
          distances <- OMstran(seqdata, indel = indel, sm = sm,
            full.matrix = full.matrix, transindel = transindel, otto = otto,
            previous = previous, add.column = add.column, with.missing=with.missing,
            weighted = weighted, refseq = refseq, norm = norm)
          result <- distances
        }
    }
  }
  # Other methods
  else {
    #  Preparation for C code
    method.num <- as.integer(method.num)
    norm.num <- as.integer(match(norm, norms[-1]) - 1) # Starts at zero
    dseqs.mat.vect <- as.integer(dseqs.mat)
    dseqs.mat.dim <- as.integer(dim(dseqs.mat))
    dseqs.lens.vect <- as.integer(dseqs.lens)

    # Dissimilarities with a reference sequence
    if (refseq.type != "none") {
      if (length(refseq.id)==1) refseq.id <- c(refseq.id,refseq.id)
      distances <- .Call(C_cstringrefseqdistance, dseqs.mat.vect, dseqs.mat.dim,
        dseqs.lens.vect, params, norm.num, method.num, as.integer(refseq.id))

      if (method %in% c("NMS", "NMSMST", "SVRspell"))
        distances <- sqrt(distances)

      if (refseq.type == "sets") {
        distances <- matrix(distances, nrow=nunique1, ncol=nunique2, byrow=FALSE)
        result <- distances[seqdata.didxs1,seqdata.didxs2,drop=FALSE]
        dimnames(result) <- list(rownames(seqdata)[refseq[[1]]],rownames(seqdata)[refseq[[2]]])
      } else {
        result <- distances[seqdata.didxs]
        # TODO Temporary fix because C++ code uses a sequence index, not a sequence object!
        names(result) <- rownames(seqdata)
        if (refseq.type == "sequence") result <- result[-length(result)]
      }

    }
    # Pairwise dissimilarities between sequences
    else {
      magic.idxs <- as.integer(c(unique(rank(seqdata.didxs, ties.method = "min")) - 1, nseqs))
      magic.seqs <- as.integer(order(seqdata.didxs))

      distances <- .Call(C_cstringdistance, dseqs.mat.vect, dseqs.mat.dim,
        dseqs.lens.vect, params, norm.num, magic.idxs, magic.seqs, method.num)

      # TODO Integrate into C++ code
      if (method %in% c("NMS", "NMSMST", "SVRspell"))
        distances <- sqrt(distances)

      # Attributes for the dist object
      class(distances) <- "dist"
      attr(distances, "Size") <- length(magic.seqs)
      attr(distances, "Labels") <- dimnames(seqdata)[[1]]
      attr(distances, "Diag") <- FALSE
      attr(distances, "Upper") <- FALSE
      attr(distances, "method") <- method

      if (full.matrix) {
        result <- dist2matrix(distances)
        dimnames(result) <- list(rownames(seqdata),rownames(seqdata))
      } else {
        result <- distances
      }
    }
  }

  #### Display elaspsed time ####

  ptime.end <- proc.time()
  time.begin <- as.POSIXct(sum(ptime.begin[1:2]), origin = "1960-01-01")
  time.end <- as.POSIXct(sum(ptime.end[1:2]), origin = "1960-01-01")
  time.elapsed <- format(round(difftime(time.end, time.begin), 3))

  msg("elapsed time:", time.elapsed)

  return(result)
}



# Adapt 'sm' for HAM (implementation requirement).
adaptSmForHAM <- function(sm, nstates, ncols) {
  costs <- array(0, dim = c(nstates, nstates, ncols))
  for (i in 1:ncols)
    costs[, , i] <- sm
  return(costs)
}
