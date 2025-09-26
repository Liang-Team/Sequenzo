seqclararange <- function(seqdata, R = 100, sample.size = 40 + 2 * max(kvals), kvals = 2:10,
                          seqdist.args = list(method = "LCS"), method = c("crisp", "fuzzy", "representativeness", "noise"), m = 1.5,
                          criteria = c("distance"), stability = FALSE, dnoise = NULL,
                          parallel = FALSE, progressbar = FALSE, keep.diss = FALSE, max.dist = NULL) {
  dlambda <- NULL ## To keep internal code
  ## Setting up and checks
  message(" [>] Starting generalized CLARA for sequence analysis.")

  if (!inherits(seqdata, "stslist")) {
    stop(" [!] seqdata should be a sequence object, see seqdef function to create one")
  }
  if (max(kvals) > sample.size) {
    stop(" [!] More clusters than the size of the sample requested.")
  }
  allmethods <- c("crisp", "fuzzy", "representativeness", "noise")
  methindex <- pmatch(method[1], allmethods)
  if (is.na(methindex)) {
    stop(" [!] Unknow method ", method, ". Please specify one of the following: ", paste(allmethods, collapse = ", "))
  }
  method <- allmethods[methindex]
  if (method == "representativeness" && is.null(max.dist)) {
    stop(" [!] You need to set max.dist to the theoretical maximum distance value when using representativeness method")
  }
  allcriteria <- c("distance", "db", "xb", "pbm", "ams")
  critindex <- pmatch(criteria, allcriteria)

  if (any(is.na(critindex))) {
    stop(" [!] At least one unkown criteria among ", paste(criteria, collapse = ", "), ". Please specify at leat one among the following: ", paste(allcriteria, collapse = ", "))
  }
  criteria <- allcriteria[critindex]
  message(" [>] Using ", method, " clustering optimizing the following criterion: ", paste(criteria, collapse = ", "), "")

  ## FIXME Add coherance check between method and criteria

  start.time <- Sys.time() # debut du processus

  # message(" [>] Overall memory requirement estimation...", appendLF=FALSE)

  # objsize <- 2*object.size(seqdata)+ R*() ## no. of column * no. of rows * 8

  ## Aggregation
  n <- nrow(seqdata)
  message(" [>] Aggregating ", n, " sequences...", appendLF = FALSE)

  ac <- wcAggregateCases(seqdata)
  agseqdata <- seqdata[ac$aggIndex, ]
  ac$probs <- ac$aggWeights / n
  message(" OK (", length(ac$aggWeights), " unique cases).")


  ## Setting up parallel computing

  if (parallel) {
    message(" [>] Initializing parallel processing...", appendLF = FALSE)
    oplan <- plan(multisession)
    on.exit(plan(oplan), add = TRUE)
    message(" OK.")
  }
  if (progressbar) {
    if (requireNamespace("progress", quietly = TRUE)) {
      old_handlers <- handlers(handler_progress(format = "(:spin) [:bar] :percent | Elapsed: :elapsed | ETA: :eta | :message"))
      if (!is.null(old_handlers)) {
        on.exit(handlers(old_handlers), add = TRUE)
      }
    } else {
      message(" [>] Install the progress package to see estimated remaining time.")
    }
    oldglobal <- handlers(global = TRUE)
    if (!is.null(oldglobal)) {
      on.exit(handlers(global = oldglobal), add = TRUE)
    }
  }
  p <- progressor(R)

  ## Memory clean up before parallel computing
  gc()
  message(" [>] Starting iterations...\n")
  ## Launch parallel loop
  # calc_pam <- foreach(loop=1:R, .export=c("davies_bouldin_internal"), .packages = c('TraMineR', 'cluster', 'WeightedCluster', 'fastcluster')) %dofuture%{#on stocke chaque sample
  calc_pam <- foreach(loop = 1:R, .options.future = list(seed = TRUE, globals = structure(TRUE, add = c("sample.size", "seqdist.args", "m", "dnoise", "dlambda")))) %dofuture% { # on stocke chaque sample
    ltime <- Sys.time()
    mysample <- sample.int(nrow(agseqdata), size = sample.size, prob = ac$probs, replace = TRUE)
    ## Re-aggregate!

    ac2 <- wcAggregateCases(data.frame(id = mysample))
    data_subset <- agseqdata[mysample[ac2$aggIndex], ]


    ## Compute distances
    seqdist.args$refseq <- NULL ## Avoid dependencies between loop
    seqdist.args$seqdata <- data_subset
    suppressMessages(diss <- do.call(seqdist, seqdist.args))
    hc <- fastcluster::hclust(as.dist(diss), method = "ward.D", members = ac2$aggWeights)


    ## For each number of clusters
    allclust <- list()
    for (k in seq_along(kvals)) {
      if (method %in% c("fuzzy", "noise")) {
        ## Weighted FCMdd clustering on subsample
        memb <- as.memb(cutree(hc, k = kvals[k]))
        algo <- ifelse(method == "fuzzy", "FCMdd", "NCdd")
        clusteringC <- wfcmdd(diss, memb = memb, weights = ac2$aggWeights, method = algo, m = m, dnoise = dnoise, dlambda=dlambda) # FCMdd algo sur la matrice de distance
        fanny <- fanny(diss, kvals[k], diss = TRUE, memb.exp = m, iniMem.p = memb, tol = 0.00001)
        clustering <- wfcmdd(diss, memb = fanny$membership, weights = ac2$aggWeights, method = algo, m = m, dnoise = dnoise, dlambda=dlambda) # FCMdd algo sur la matrice de distance
        if (clusteringC$functional < clustering$functional) {
          clustering <- clusteringC
        }
        rm(fanny, clusteringC)
        ## Retrieve medoids
        if(!is.null(dlambda)){
        	dnoise <- clustering$dnoise
        }
        medoids <- mysample[ac2$aggIndex[clustering$mobileCenters]] ## Going back to overall dataset
      } else {
        ## Weighted Pam clustering on subsample
        clustering <- wcKMedoids(diss, k = kvals[k], cluster.only = TRUE, initialclust = hc, weights = ac2$aggWeights) # PAM sur la matrice de distance
        ## Retrieve medoids
        medoids <- mysample[ac2$aggIndex[unique(clustering)]] ## Going back to overall dataset
      }
      rm(clustering)
      ## Compute distances between all sequence to the medoids
      distargs2 <- seqdist.args
      distargs2$seqdata <- agseqdata
      distargs2$refseq <- list(1:nrow(agseqdata), medoids)
      suppressMessages(diss2 <- do.call(seqdist, distargs2))
      ## Compute two minimal distances are used for silhouette width
      ## and other criterions
      alphabeta <- apply(diss2, 1, function(x) sort(x)[1:2])
      # alpha <- alphabeta[1, ]
      # beta <- alphabeta[2, ]
      sil <- ((alphabeta[2, ] - alphabeta[1, ]) / pmax(alphabeta[2, ], alphabeta[1, ]))
      if (method == "fuzzy") {
        ## Allocate to clusters using FCM formulae
        memb <- (1 / diss2)^(1 / (m - 1))
        memb <- memb / rowSums(memb)
        memb[diss2 == 0] <- 1
        ## Compute criterion (FCMdd Formulae)
        ## mean_diss <- sum(rowSums((memb^m)*diss2)*ac$aggWeights)
        mean_diss <- sum(rowSums((memb^m) * diss2) * ac$probs)

        db <- fuzzy_davies_bouldin_internal(diss2, memb, medoids, weights = ac$aggWeights)$db

        alpha <- 1
        hightest.memb <- apply(memb, 1, function(x) {
          y <- sort(x, decreasing = TRUE)[1:2]
          return((y[1] - y[2])**alpha)
        })
        pbm <- ((1 / length(medoids)) * (max(diss2[medoids, ]) / sum(rowSums((memb) * diss2) * ac$probs)))^2
        ams <- sum(hightest.memb * sil * ac$probs) / sum(hightest.memb * ac$probs)
        rm(hightest.memb)
      } else if (method == "noise") {
        ## Allocate to clusters using FCM formulae
        diss3 <- cbind(diss2, dnoise)
        memb <- (1 / diss3)^(1 / (m - 1))
        memb <- memb / rowSums(memb)
        memb[diss3 == 0] <- 1
        ## Compute criterion (FCMdd Formulae)
        ## mean_diss <- sum(rowSums((memb^m)*diss2)*ac$aggWeights)
        mean_diss <- sum(rowSums((memb^m) * diss3) * ac$probs)

        db <- fuzzy_davies_bouldin_internal(diss2, memb[, -ncol(memb), drop = FALSE], medoids, weights = ac$aggWeights)$db

        alpha <- 1
        hightest.memb <- apply(memb[, -ncol(memb), drop = FALSE], 1, function(x) {
          y <- sort(x, decreasing = TRUE)[1:2]
          return((y[1] - y[2])**alpha)
        })
        pbm <- ((1 / length(medoids)) * (max(diss2[medoids, ]) / sum(rowSums((memb) * diss3) * ac$probs)))^2
        ams <- sum(hightest.memb * sil * ac$probs) / sum(hightest.memb * ac$probs)
        rm(hightest.memb)
      } else {
        ## Allocate to clusters
        memb <- apply(diss2, 1, which.min)

        mean_diss <- sum(alphabeta[1, ] * ac$probs)
        db <- davies_bouldin_internal(diss2, memb, medoids, weights = ac$aggWeights)$db
        pbm <- ((1 / length(medoids)) * (max(diss2[medoids, ]) / mean_diss))^2
        ams <- sum(sil * ac$probs)
      }


      distmed <- as.dist(diss2[medoids, ])
      minsep <- min(distmed)
      maxsep <- max(distmed)
      xb <- mean_diss / minsep
      ## Memory cleanup
      rm(alphabeta, sil, diss2)
      rm(distmed, minsep, maxsep)
      ## Store results
      ## Do not disagg Now to save memory
      allclust[[k]] <- list(mean_diss = mean_diss, db = db, pbm = pbm, ams = ams, xb = xb, clustering = memb, medoids = medoids)
    }

    ## Store the clustering of this iteration to stratify the next one
    # previousclusteringstrata <- allclust[[length(kvals)]]$clustering
    rm(diss)
    gc()
    # p(message=paste("Iteration ", loop, " finished. Objective=", round(mean_diss, 2), ". Time=", format(round(Sys.time()-ltime, 2)), ".\n"))
    p()
    # Return all clusterings
    allclust
  }
  ## Stop parallel computing
  message("\n [>] Aggregating iterations for each k values...")
  flush.console()
  ## Function to access data from parallel computing
  reframeData <- function(ll, name, clust, format = "vector") {
    xx <- lapply(ll, FUN = function(x) x[[clust]][[name]])
    if (format == "list") {
      return(xx)
    } else if (format == "matrix") {
      return(do.call(cbind, xx))
    }
    return(do.call(c, xx))
  }

  kvalscriteria <- expand.grid(kvals = seq_along(kvals), criteria = criteria)
  ## Object to be returned, should match clustrange structure
  p <- progressor(nrow(kvalscriteria) * R)
  kret <- list()
  for (i in 1:nrow(kvalscriteria)) {
    k <- kvalscriteria$kvals[i]
    criteria <- as.character(kvalscriteria$criteria[i])
    ## Objective criteria
    mean_all_diss <- reframeData(calc_pam, "mean_diss", k, "vector")
    pbm_all <- reframeData(calc_pam, "pbm", k, "vector")
    db_all <- reframeData(calc_pam, "db", k, "vector")
    xb_all <- reframeData(calc_pam, "xb", k, "vector")
    ams_all <- reframeData(calc_pam, "ams", k, "vector")
    ## Retrieve all clusterings
    clustering_all_diss <- reframeData(calc_pam, "clustering", k, ifelse(method %in% c("fuzzy", "noise"), "list", "matrix"))
    ## Retrieve medoids
    med_all_diss <- reframeData(calc_pam, "medoids", k, "matrix")
    ## Find best clustering
    objective <- switch(criteria,
      distance = mean_all_diss,
      pbm = pbm_all,
      db = db_all,
      ams = ams_all,
      xb = xb_all
    )
    best <- ifelse(criteria %in% c("ams", "pbm"), which.max(objective), which.min(objective))

    ## Compute clustering stability of the best partition
    # ari <- sapply(1:ncol(clustering_all_diss), function(x){
    # tab <- xtabs(ac$aggWeights~clustering_all_diss[, best]+clustering_all_diss[, x])
    # adjustedRandIndex(tab)
    # })
    if (stability) {
      if (method %in% c("noise", "fuzzy")) {
        foplan <- plan(sequential)
      }
      j <- "BindingFIX"
      arilist <- foreach(j = 1:R, .options.future = list(seed = TRUE, globals = structure(TRUE, add = c("ac", "clustering_all_diss", "method")))) %dofuture% { # on stocke chaque sample

        if (method %in% c("noise", "fuzzy")) {
          tab <- as.table(crossmemb(clustering_all_diss[[j]] * ac$aggWeights, clustering_all_diss[[best]] * ac$aggWeights, relativize = FALSE))
        } else {
          tab <- as.table(tapply(ac$aggWeights, list(clustering_all_diss[, j], clustering_all_diss[, best]), sum, default = 0L))
        }
        val <- c(adjustedRandIndex(tab), jaccardCoef(tab))
        p()
        val
      }
      rm(j)
      if (method %in% c("noise", "fuzzy")) {
        plan(foplan)
      }
      arimatrix <- do.call(rbind, arilist)
      colnames(arimatrix) <- c("ARI", "JC")
      ari08 <- sum(arimatrix[, 1] >= 0.8)
      jc08 <- sum(arimatrix[, 2] >= 0.8)
    } else {
      arimatrix <- NA
      ari08 <- NA
      jc08 <- NA
    }
    if (method %in% c("noise", "fuzzy")) {
      disagclust <- clustering_all_diss[[best]][ac$disaggIndex, ] ## Disaggregate here
    } else {
      disagclust <- clustering_all_diss[ac$disaggIndex, best] ## Disaggregate here
    }
    if (criteria %in% c("ams", "pbm")) {
      evol.diss <- cummax(objective)
    } else {
      evol.diss <- cummin(objective)
    }

    ## Store the best solution and evaluations of the others.
    bestcluster <- list(
      medoids = ac$aggIndex[med_all_diss[, best]], ## Disaggregate here
      # clustering = clustering_all_diss[, best],
      clustering = disagclust, ## Disaggregate here
      evol.diss = evol.diss,
      iter.objective = objective,
      objective = objective[best],
      iteration = best,
      # ari=ari,
      # iter.ari09=iter.ari,
      arimatrix = arimatrix,
      criteria = criteria,
      method = method,
      avg_dist = mean_all_diss[best],
      pbm = pbm_all[best],
      db = db_all[best],
      xb = xb_all[best],
      ams = ams_all[best],
      ari08 = ari08,
      jc08 = jc08,
      R = R,
      k = k
    ) # Creating the object to be returned

    #### Compute cluster quality David Bouldin if asked for

    if (keep.diss || method == "representativeness") {
      ## Recompute distances (required to avoid memory issue)
      seqdist.args$seqdata <- agseqdata
      seqdist.args$refseq <- list(1:nrow(agseqdata), ac$disaggIndex[bestcluster$medoids])
      suppressMessages(diss2 <- do.call(seqdist, seqdist.args))
      bestcluster$diss <- diss2[ac$disaggIndex, ]
      bestcluster$representativeness <- 1 - bestcluster$diss / max.dist
      rm(diss2)
    }

    ## Store computed cluster quality
    gc()
    class(bestcluster) <- "seqclara"
    kresult <- list(k = k, criteria = criteria, stats = c(bestcluster$avg_dist, bestcluster$pbm, bestcluster$db, bestcluster$xb, bestcluster$ams, bestcluster$ari08, bestcluster$jc08, best), bestcluster = bestcluster)
    kret[[i]] <- kresult
  }
  claraObj <- function(kretlines) {
    if (method == "crisp") {
      clustering <- matrix(-1, nrow = nrow(seqdata), ncol = length(kvals))
      clustering <- as.data.frame(clustering)
      colnames(clustering) <- paste("cluster", kvals, sep = "")
    } else {
      clustering <- list()
    }
    ret <- list(
      kvals = kvals,
      clara = list(),
      clustering = clustering,
      stats = matrix(-1, nrow = length(kvals), ncol = 8)
    )

    for (i in kretlines) {
      k <- kret[[i]]$k
      criteria <- kret[[i]]$criteria
      ret$stats[k, ] <- kret[[i]]$stats
      ret$clara[[k]] <- kret[[i]]$bestcluster
      if (method == "crisp") {
        ret$clustering[, k] <- kret[[i]]$bestcluster$clustering
      } else if (method == "representativeness") {
        ret$clustering[[paste0("cluster", kvals[k])]] <- kret[[i]]$bestcluster$representativeness
      } else {
        ret$clustering[[paste0("cluster", kvals[k])]] <- kret[[i]]$bestcluster$clustering
      }
    }
    rownames(ret$stats) <- paste("cluster", kvals, sep = "")
    colnames(ret$stats) <- c("Avg dist", "PBM", "DB", "XB", "AMS", "ARI>0.8", "JC>0.8", "Best iter")
    class(ret) <- c("seqclararange", "clustrange")
    return(ret)
  }
  if (length(criteria) > 1) {
    ret <- list(param = list(criteria = criteria, pam.combine = FALSE, all.criterias = criteria, kvals = kvals, method = method, stability = stability))

    for (meth in criteria) {
      ret[[meth]] <- claraObj(which(kvalscriteria$criteria == meth))
    }

    allstats <- list()
    for (meth in criteria) {
      allstats[[meth]] <- as.data.frame(cbind(ret[[meth]]$stats, ngroup = kvals))
      allstats[[meth]]$criteria <- meth
    }
    ret$allstats <- do.call(rbind, allstats)
    class(ret) <- c("seqclarafamily", "clustrangefamily")
  } else {
    ret <- claraObj(1:nrow(kvalscriteria))
  }

  ## Overall memory management
  gc()

  message(" \n [>] Overall computation time : ", format(round(Sys.time() - start.time, 2)))
  return(ret)
}


# mysample <- strataSampling(precluststrata, sample.size=sample.size, prob=ac$probs, randomize=0)


adjustedRandIndex <- function(x, y = NULL) {
  if (!is.table(x)) {
    x <- as.vector(x)
    y <- as.vector(y)
    if (length(x) != length(y)) {
      stop("arguments must be vectors of the same length")
    }
    tab <- table(x, y)
  } else {
    tab <- x
  }
  if (all(dim(tab) == c(1, 1))) {
    return(1)
  }
  a <- sum(choose(tab, 2))
  b <- sum(choose(rowSums(tab), 2)) - a
  c <- sum(choose(colSums(tab), 2)) - a
  d <- choose(sum(tab), 2) - a - b - c
  ARI <- (a - (a + b) * (a + c) / (a + b + c + d)) / ((a + b +
    a + c) / 2 - (a + b) * (a + c) / (a + b + c + d))
  return(ARI)
}


jaccardCoef <- function(tab) {
  if (all(dim(tab) == c(1, 1))) {
    return(1)
  }
  n11 <- sum(tab^2)
  n01 <- sum(colSums(tab)^2)
  n10 <- sum(rowSums(tab)^2)

  return(n11 / (n01 + n10 - n11))
}




daviesBouldinIndex <- function(seqclara, seqdata, seqdist.args = list(method = "LCS"), p = 1) {
  ret <- numeric(length(seqclara$kvals))
  for (k in seq_along(seqclara$kvals)) {
    ## Objective criteria

    seqdist.args$seqdata <- seqdata
    seqdist.args$refseq <- list(1:nrow(seqdata), seqclara$clara[[k]]$medoids)
    suppressMessages(diss2 <- do.call(seqdist, seqdist.args))
    db <- davies_bouldin_internal(diss2, seqclara$clara[[k]]$clustering, seqclara$clara[[k]]$medoids)$db
    rm(diss2)
    ret[k] <- db
  }
  return(ret)
}
