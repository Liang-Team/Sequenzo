checkcost <- function(sma, seqdata, with.missing, indel, tol = NULL) {

  msg("checking 'sm' (size and triangle inequality)")

  if (is.null(tol)) {
    tol <- 1e-07 * max(sma)
  }

  alphabet <- attr(seqdata, "alphabet")
  ## Gaps in sequences
  if (with.missing) {
    alphabet <- c(alphabet, attr(seqdata, "nr"))
  }
  alphsize <- length(alphabet)

  if (length(dim(sma)) == 2) {
    dim(sma) <- c(dim(sma), 1)
  } else {
    if (ncol(seqdata) != dim(sma)[3]) {
      stop(" [!] size of substitution cost matrix must be ", alphsize, "x",
        alphsize, "x", ncol(seqdata), call. = FALSE)
    }
  }

  if (!missing(indel) && any(indel <= 0)) {
    stop(" [!] indel cost should be positive")
  }

  for (i in 1:dim(sma)[3]) {
    sm <- sma[, , i]
    if (nrow(sm) != alphsize | ncol(sm) != alphsize) {
      stop(" [!] size of substitution cost matrix must be ", alphsize, "x",
        alphsize, call. = FALSE)
    }

    if (any(sm < 0)) {
      stop(" [!] Negative substitution costs are not allowed", call. = FALSE)
    }

    if (any(diag(sm) != 0)) {
      stop(" [!] All element on the diagonal of sm (substitution cost) should be equal to zero")
    }

    triangleineq <- checktriangleineq(sm, warn = FALSE, indices = TRUE, tol = tol)

    ## triangleineq contains a vector of problematic indices.
    if (!is.logical(triangleineq)) {
      message(" [!!] at least, one substitution cost doesn't respect the triangle inequality.\n",
        " [!!] replacing ", alphabet[triangleineq[1]], " with ", alphabet[triangleineq[3]],
        " (cost=", format(sm[triangleineq[1], triangleineq[3]]), ") and then ",
        alphabet[triangleineq[3]], " with ", alphabet[triangleineq[2]], " (cost=",
        format(sm[triangleineq[3], triangleineq[2]]), ")\n [!] costs less than replacing directly ",
        alphabet[triangleineq[1]], " with ", alphabet[triangleineq[2]], " (cost=",
        format(sm[triangleineq[1], triangleineq[2]]), ")\n", " [!] total difference ([",
        alphabet[triangleineq[1]], "=>", alphabet[triangleineq[3]], "] + [",
        alphabet[triangleineq[3]], "=>", alphabet[triangleineq[2]], "] - [",
        alphabet[triangleineq[1]], "=>", alphabet[triangleineq[2]], "]): ",
        format(sm[triangleineq[1], triangleineq[3]] + sm[triangleineq[3],
          triangleineq[2]] - sm[triangleineq[1], triangleineq[2]]))
    }

    if (!missing(indel)) {
      if (length(indel) > 1) {
        if (length(indel) != alphsize) {
          stop(" [!] You should specify either one global indel or one indel per state.")
        }
        sm2 <- cbind(sm, indel)
        sm2 <- rbind(sm2, c(indel, 0))
        triangleineq <- checktriangleineq(sm2, warn = FALSE, indices = TRUE, tol = tol)
        ## triangleineq contains a vector of problematic indices.
        if (!is.logical(triangleineq)) {
          warning(" [!] at least, one indel cost does not respect the triangle inequality.\n",
          call. = FALSE)
        }
      } else if (any(sm > 2 * indel)) {
        warning("At least one substitution cost greater than twice the indel cost.",
          " Such substitution costs will never be used.", call. = FALSE)
      }
    }

    ## Testing for symmetric matrix
    if (any((sm - t(sm)) > tol)) {
      warning("The substitution cost matrix is not symmetric.", call. = FALSE)
    }
  }
}
