# mysample <- strataSampling(precluststrata, sample.size=sample.size, prob=ac$probs, randomize=0)






bootclustrange <- function(object, seqdata, seqdist.args = list(method = "LCS"),
                           R = 100, sample.size = 1000,
                           parallel = FALSE, progressbar = FALSE,
                           sampling = "clustering", strata = NULL) {
  ## Preparing
  start.time <- Sys.time() # debut du processus
  if (inherits(object, "seqclararange")) {
    medoids <- object$clara[[length(object$clara)]]$medoids
    clustering <- as.data.frame(object$clustering)
  } else {
    clustering <- as.data.frame(object)
    medoids <- NULL
  }
  if (is.null(strata) && sampling == "clustering") {
    strata <- clustering[, ncol(clustering)]
    tt <- prop.table(table(strata))
    if (any(round(tt * sample.size) < 2)) {
      stop("[!] sample size is to small for the stratified sampling of clustering. Consider a minimum value of ", 2 / min(tt))
    }
    sampling <- "strata"
  }
  if (parallel) {
    oplan <- plan(multisession)
    on.exit(plan(oplan), add = TRUE)
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

  ## Parallel loop
  calc_cqi <- foreach(loop = 1:R, .options.future = list(packages = c("TraMineR", "WeightedCluster"), seed = TRUE, globals = structure(TRUE, add = c("sample.size", "seqdist.args")))) %dofuture% { # on stocke chaque bootstrap

    ## function for stratified subsampling
    stratsampling <- function() {
      tt <- table(strata)
      tosample <- round(prop.table(tt) * sample.size)
      correction <- sample.size - sum(tosample)
      if (correction != 0) {
        correct <- sample(1:length(tosample), abs(correction))
        tosample[correct] <- tosample[correct] + sign(correction)
      }
      mysample <- NULL
      for (u in unique(strata)) {
        mysample <- c(mysample, sample(which(strata == u), tosample[as.character(u)]))
      }
      return(mysample)
    }
    ltime <- proc.time()

    ## Ensure that we have at least one observation per cluster in our bootstrap
    done <- FALSE
    while (!done) {
      if (sampling == "strata") {
        mysample <- stratsampling()
      } else if (sampling == "medoids" && !is.null(medoids)) {
        mysample <- c(medoids, sample.int(nrow(clustering), sample.size - length(medoids)))
      } else {
        mysample <- sample.int(nrow(clustering), sample.size)
      }
      clust_sample <- clustering[mysample, ]
      done <- all(sapply(1:ncol(clustering), function(x) length(unique(clustering[, x])) == length(unique(clust_sample[, x]))))
    }

    ## Subsample
    seqdist.args$seqdata <- seqdata[mysample, ]
    suppressMessages(diss <- do.call(seqdist, seqdist.args))
    cqi <- as.clustrange(clustering[mysample, ], diss = diss)
    rm(diss)
    gc()
    cqi$stats
  }

  reframeData <- function(ll, clust, matrix = FALSE) {
    xx <- lapply(ll, FUN = function(x) x[clust, ])
    if (matrix) {
      return(do.call(rbind, xx))
    }
    return(do.call(c, xx))
  }

  ## Build the corresponding clustrange object.
  ret <- list(clustering = clustering)
  numclust <- ncol(ret$clustering)
  ret$kvals <- numeric(numclust)

  ret$meant <- matrix(-1, nrow = numclust, ncol = 10)
  ret$stderr <- matrix(-1, nrow = numclust, ncol = 10)
  ret$boot <- list()
  for (i in 1:numclust) {
    ret$boot[[i]] <- reframeData(calc_cqi, i, TRUE)
    ret$kvals[i] <- length(unique(ret$clustering[, i]))
    ret$meant[i, ] <- colMeans(ret$boot[[i]])
    ret$stderr[i, ] <- apply(ret$boot[[i]], 2L, function(x) sqrt(var(x)))
  }
  clnames <- paste0("cluster", ret$kvals)
  colnames(ret$clustering) <- clnames
  ret$meant <- as.data.frame(ret$meant)
  ret$stderr <- as.data.frame(ret$stderr)
  colnames(ret$meant) <- colnames(ret$boot[[i]])
  colnames(ret$stderr) <- colnames(ret$boot[[i]])
  rownames(ret$meant) <- clnames
  rownames(ret$stderr) <- clnames
  ret$stats <- ret$meant
  class(ret) <- c("bootclustrange", "clustrange", class(ret))
  return(ret)
}

plot.bootclustrange <- function(x, stat = "noCH", legendpos = "bottomright", norm = "none",
                                withlegend = TRUE, lwd = 1, col = NULL, ylab = "Indicators",
                                xlab = "N clusters", conf.int = 0.95, ci.method = "perc", ci.alpha = .3, line = "median", ...) {
  # 	NextMethod("plot")
  getS3method("plot", "clustrange")(x, stat = stat, legendpos = legendpos, norm = norm,
    withlegend = withlegend, lwd = lwd, col = col, ylab = ylab,
    xlab = xlab, conf.int = conf.int, ci.method = ci.method, ci.alpha = ci.alpha, line = line, ...)
}

print.bootclustrange <- function(x, digits = 2, bootstat = c("mean"), ...) {
  getS3method("print", "clustrange")(x, digits = digits, bootstat = bootstat, ...)
}
