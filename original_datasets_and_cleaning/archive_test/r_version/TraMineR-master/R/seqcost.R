# Compute substitution costs and substitution-cost/proximity matrix

seqcost <- function(seqdata, method, cval = NULL, with.missing = FALSE,
  miss.cost = NULL, time.varying = FALSE, weighted = TRUE, transition = "both",
  lag = 1, miss.cost.fixed = NULL, state.features = NULL, feature.weights = NULL,
  feature.type = list(), proximities = FALSE) {

  if (!inherits(seqdata, "stslist")) {
    stop(" [!] data is NOT a sequence object, see seqdef function to create one")
  }
  metlist <- c("CONSTANT", "TRATE", "FEATURES", "FUTURE", "INDELS", "INDELSLOG")
  if (missing(method) || !method %in% metlist) {
    stop(" [!] method must be one of: ", paste(metlist, collapse = " "))
  }

  transitionlist <- c("previous", "next", "both")
  if (!transition %in% transitionlist) {
    stop(" [!] transition must be one of: ", paste(transitionlist, collapse = " "))
  }

  if (isTRUE(with.missing) && !any(seqdata==attr(seqdata,"nr"))) {
    with.missing <- FALSE
    msg.warn("seqcost: 'with.missing' set as FALSE as 'seqdata' has no non-void missing values")
  }

  ret <- list()
  ret$indel <- 1
  alphabet <- attr(seqdata, "alphabet")

  cval4cond <- time.varying && method == "TRATE" && transition == "both"
  if (is.null(cval)) {
    cval <- ifelse(cval4cond, 4, 2)
  }
  if (is.null(miss.cost)) {
    miss.cost <- cval
  }
  if (is.null(miss.cost.fixed)){
    if (method %in% c("INDELS","INDELSLOG")) {
      miss.cost.fixed <- FALSE
    } else {
      miss.cost.fixed <- TRUE
    }
  }
  ## Adding an entry for for missing state
  if (with.missing) {
    ## if(!miss.cost.fixed && method %in% c('TRATE','FUTURE','INDELS','INDELSLOG') {
    ## message(' [>] Substitution costs for missing values derived from data') } else
    ## { message(' [>] Substitution costs for missing values set as ',miss.cost) }

    alphabet <- c(alphabet, attr(seqdata, "nr"))
  }

  alphsize <- length(alphabet)

  if (method == "CONSTANT") {
    if (is.na(cval)) {
      stop("no value for the constant substitution-cost")
    }
    if (time.varying) {
      time <- ncol(seqdata)

      message(" [>] creating ", alphsize, "x", alphsize, "x", time, " time varying substitution-cost matrix using ",
        cval, " as constant value")
      costs <- array(cval, dim = c(alphsize, alphsize, time))
      for (i in 1:time) {
        diag(costs[, , i]) <- 0
      }
    } else {
      message(" [>] creating ", alphsize, "x", alphsize, " substitution-cost matrix using ",
        cval, " as constant value")

      costs <- matrix(cval, nrow = alphsize, ncol = alphsize)
      diag(costs) <- 0
    }
  }
  if (method == "FUTURE") {

    chisqdista <- function(mat) {
      cs <- colSums(mat)
      if (any(cs == 0)) {
        cs[cs == 0] <- Inf
      }
      Pdot <- 1/cs
      n <- nrow(mat)
      dist <- matrix(0, nrow = n, ncol = n)
      for (i in 1:n) {
        if (i < n) {
          for (j in (i + 1):n) {
          dist[i, j] = sum(Pdot * (mat[i, ] - mat[j, ])^2)
          dist[j, i] = dist[i, j]
          }
        }
      }
      return(sqrt(dist))
    }

    if (time.varying) {
      stop(" [!] time.varying substitution cost is not (yet) implemented for method FUTURE.")
    }
    message(" [>] creating substitution-cost matrix using common future...")
    tr <- seqtrate(seqdata, time.varying = FALSE, weighted = weighted, lag = lag,
      with.missing = with.missing)
    costs <- chisqdista(tr)
    diag(costs) <- 0
    ret$indel <- 0.5 * max(costs)
  }

  if (method == "FEATURES") {
    if (time.varying) {
      stop(" [!] time.varying substitution cost is not (yet) implemented for method FEATURES.")
    }
    if (is.null(state.features) || nrow(state.features) != length(alphabet)) {
      stop(" [!] state.features should be a data.frame containing one row per state (possibly one for missing values).")
    }
    if (!requireNamespace("cluster")) {
      stop(" [!] cluster library is required to use FEATURES method.")
    }
    if (is.null(feature.weights)) {
      feature.weights <- rep(1, ncol(state.features))
    }
    costs <- as.matrix(daisy(state.features, metric = "gower", weights = feature.weights,
      type = feature.type))
    diag(costs) <- 0
    ret$indel <- 0.5 * max(costs)
  }
  if (method == "INDELS" || method == "INDELSLOG") {
    if (time.varying) {
      #stop(" [!] time.varying substitution cost is not (yet) implemented for INDELS and INDELSLOG.")
      indels <- seqstatd(seqdata, with.missing=with.missing)$Frequencies
    } else {
      ww <- attr(seqdata, "weights")
      if (is.null(ww) || !weighted) {
        ww <- rep(1, nrow(seqdata))
      }
      indels <- as.numeric(prop.table(xtabs(rep(ww, ncol(seqdata)) ~ unlist(seqdata)))[as.character(alphabet)])
    }

    #ww <- attr(seqdata, "weights")
    #if (is.null(ww) || !weighted) {
    #  ww <- rep(1, nrow(seqdata))
    #}
    #indels <- as.numeric(prop.table(xtabs(rep(ww, ncol(seqdata)) ~ unlist(seqdata)))[alphabet])
    indels[is.na(indels)] <- 1
    if (method == "INDELSLOG") {
      indels <- log(2/(1 + indels))
    } else {
      indels <- 1/indels
      indels[is.infinite(indels)] <- .Machine$double.xmax
    }
    ## ret <- list()
    ret$indel <- indels
    ## ret$sm <- matrix(0, nrow=length(alphabet), ncol=length(alphabet))

    if (time.varying) {
      time <- ncol(seqdata)
      message(" [>] creating ", alphsize, "x", alphsize, "x", time, " time varying substitution-cost matrix using ",method)
      costs <- array(0, dim = c(alphsize, alphsize, time))
      for (t in 1:time) {
        for (i in seq_along(alphabet)) {
          for (j in seq_along(alphabet)) {
            if (i != j) {
              ## ret$sm[i, j] <- indels[i]+indels[j]
              costs[i, j, t] <- indels[i, t] + indels[j, t]
            }
          }
        }
      }
    } else {
      costs <- matrix(0, nrow = length(alphabet), ncol = length(alphabet))
      for (i in seq_along(alphabet)) {
        for (j in seq_along(alphabet)) {
          if (i != j) {
            ## ret$sm[i, j] <- indels[i]+indels[j]
            costs[i, j] <- indels[i] + indels[j]
          }
        }
      }
    }
    costs[is.infinite(ret$sm)] <- .Machine$double.xmax
    ## ret$sm[is.infinite(ret$sm)] <- .Machine$double.xmax return(ret)
  }
  if (method == "TRATE") {
    if (time.varying) {
      message(" [>] creating time varying substitution-cost matrix using transition rates ...")
      tr <- seqtrate(seqdata, time.varying = TRUE, weighted = weighted, lag = lag,
        with.missing = with.missing)
      tmat <- nrow(tr)
      time <- ncol(seqdata)
      costs <- array(0, dim = c(alphsize, alphsize, time))
      ## Function to compute the cost according to transition rates
      tratecostBoth <- function(trate, time, state1, state2, debut, fin) {
        cost <- 0
        if (!debut) {
          ## Premier état
          cost <- cost - trate[state1, state2, time - 1] - trate[state2,
          state1, time - 1]
        }
        if (!fin) {
          ## Dernier Etat
          cost <- cost - trate[state1, state2, time] - trate[state2, state1,
          time]
        }
        if (!debut && !fin) {
          return(cost + cval)
        } else {
          return(cval + 2 * cost)
        }
      }
      tratecostPrevious <- function(trate, time, state1, state2, debut, fin) {
        cost <- 0
        if (!debut) {
          ## Premier état
          cost <- cost - trate[state1, state2, time - 1] - trate[state2,
          state1, time - 1]
        }
        return(cval + cost)
      }
      tratecostNext <- function(trate, time, state1, state2, debut, fin) {
        cost <- 0
        if (!fin) {
          ## Dernier Etat
          cost <- cost - trate[state1, state2, time] - trate[state2, state1,
          time]
        }
        return(cval + cost)
      }
      if (transition == "previous") {
        tratecost <- tratecostPrevious
      } else if (transition == "next") {
        tratecost <- tratecostNext
      } else {
        tratecost <- tratecostBoth
      }

      for (t in 1:time) {
        for (i in 1:(tmat - 1)) {
          for (j in (i + 1):tmat) {
          cost <- max(0, tratecost(tr, t, i, j, t == 1, t == time))
          costs[i, j, t] <- cost
          costs[j, i, t] <- cost
          }
        }
      }
    } else {
      message(" [>] creating substitution-cost matrix using transition rates ...")
      tr <- seqtrate(seqdata, time.varying = FALSE, weighted = weighted, lag = lag,
        with.missing = with.missing)
      tmat <- nrow(tr)
      costs <- matrix(nrow = alphsize, ncol = alphsize)
      diag(costs) <- 0
      for (i in 1:(tmat - 1)) {
        for (j in (i + 1):tmat) {
          cost <- cval - tr[i, j] - tr[j, i]
          costs[i, j] <- cost
          costs[j, i] <- cost
        }
      }
      ret$indel <- 0.5 * max(costs)
    }
  }

  if (with.missing && miss.cost.fixed) {
    if (time.varying) {
      costs[alphsize, 1:(alphsize - 1), ] <- miss.cost
      costs[1:(alphsize - 1), alphsize, ] <- miss.cost
    } else {
      costs[alphsize, 1:(alphsize - 1)] <- miss.cost
      costs[1:(alphsize - 1), alphsize] <- miss.cost
    }
  }

  ## Setting rows and columns labels
  #rclab <- paste(alphabet, "->", sep = "")
  rclab <- alphabet
  if (time.varying) {
    dimnames(costs) <- list(rclab, rclab, colnames(seqdata))
  } else {
    dimnames(costs) <- list(rclab, rclab)
  }
  if (proximities)
    ret$prox <- 1 - costs / max(costs)
  else
    ret$sm <- costs
  return(ret)
}

## alias for backward compatibility with seqsubm
seqsubm <- function(...) {
  return(seqcost(...)$sm)
}
