#' Measures of association between domains of sequence data
#'
#' The function computes pairwise domain association based on cross-tabulation of the states observed in the sequences of the two domains involved. The association measure returned can be Cramer's V or the likelihood ratio (LRT).
#'
#' @details
#' For each pair of domains, \code{seqdomassoc} cross-tabulates the position-wise states across domains using all sequences when \code{rep.method = "overall"}.  When \code{rep.method = "rep"}, each observed sequence is first replaced by the closest representative sequence and, when \code{rep.method = "eq.group"}, each observed sequence is replaced by the group medoid of its group. Then, the selected association measures are computed on the resulting cross-tables.
#'
#' The \code{"overall"} method implies a strong position-wise association and will not detect association occurring after a small time warp. With representative sequences, the same holds, but for representatives only. Using dissimilarity measures that allow for time warp for identifying representatives, observed sequences may differ from their representatives in the timing of the states. Therefore, using representatives instead of all sequences relaxes somewhat the strong timing constraint.
#'
#' @param seqdata.dom List of \code{stslist} objects (one per dimension)
#' @param diss.dom List of dissimilarity matrices used for selecting representatives. Ignored when \code{rep.method="overall"}.
#' @param assoc Character string. The association measure to be computed. One of "V" (Cramer V) or "LRT" or a vector with both.
#' @param rep.method Character string. Method for determining the sequences on which the association is computed. One of "rep" (representative sequences), "eq.group" (medoids of equally spaced groups), or "overall".
#' @param wrange Vector of two integers. Window range for count of co-occurrences. A state at \code{p} in the first domain is compared with states in [\code{p+wrange[1]}, \code{p+wrange[2]}] in the second domain.
#' @param p.value Logical. Should p-values be returned?
#' @param struct.zero Logical. Should zeros in cross tables be treated as structural zeros?
#' @param cross.table Logical. Should cross tables be returned? If \code{TRUE}, cross tables are returned as the list attribute \code{cross.tables}.
#' @param with.missing Logical. Should missing be treated as a regular state.
#' @param weighted Logical. Should sequence weights be taken into account when present in the sequence objects? When applicable, weights of the first domain are used.
#' @param seqrep.args List of arguments passed to \code{\link[TraMineR]{seqrep}} when \code{rep.method="rep"}.
#' @param seqrf.args List of arguments passed to \code{\link[TraMineR]{seqrf}} when \code{rep.method="eq.group"}.
#' @param dnames String vector: names of dimensions.
#'
#' @return A matrix with the list of cross tables in attribute \code{cross.tables}.
#'
#' @references
#' Ritschard, G., T.F. Liao, and E. Struffolino (2023). Strategies for
#' multidomain sequence analysis in social research.
#' \emph{Sociological Methodology}, in press.
#'
#' @seealso \code{\link{print.sdomassoc}}
#'
#' @author Gilbert Ritschard
#'
#' @export
#'
#' @exportS3Method print sdomassoc
#'
#' @import TraMineR
#' @importFrom stats loglin xtabs pchisq
#'
#' @examples
#' data(biofam)
#'
#' ## Building one channel per type of event (left, children or married)
#' cases <- 1:50
#' bf <- as.matrix(biofam[cases, 10:25])
#' children <-  bf==4 | bf==5 | bf==6
#' married <- bf == 2 | bf== 3 | bf==6
#' left <- bf==1 | bf==3 | bf==5 | bf==6
#'
#' ## Building sequence objects
#' child.seq <- seqdef(children, weights = biofam[cases,'wp00tbgs'])
#' marr.seq <- seqdef(married, weights = biofam[cases,'wp00tbgs'])
#' left.seq <- seqdef(left, weights = biofam[cases,'wp00tbgs'])
#'
#' ## distances by channel
#' dchild <- seqdist(child.seq, method="OM", sm="INDELSLOG")
#' dmarr <- seqdist(marr.seq, method="OM", sm="INDELSLOG")
#' dleft <- seqdist(left.seq, method="OM", sm="INDELSLOG")
#' dbiofam <- list(dchild,dmarr,dleft)
#' dnames <- names(dbiofam) <- c("child","marr","left")
#'
#'
#' seqdomassoc(list(child.seq,marr.seq,left.seq), dnames=dnames)
#' seqdomassoc(list(child.seq,marr.seq,left.seq), diss.dom=dbiofam,
#'             rep.method="rep", assoc="V", dnames=dnames)
#' seqdomassoc(list(child.seq,marr.seq,left.seq), diss.dom=dbiofam,
#'             rep.method="eq.group", assoc="V", dnames=dnames)
#'
#'


seqdomassoc <- function(seqdata.dom, rep.method="overall", assoc = c("LRT","V"),
                        diss.dom=NULL, wrange=NULL,
                        p.value=TRUE, struct.zero=TRUE,
                        cross.table=FALSE,
                        with.missing=FALSE, weighted=TRUE,
                        seqrep.args=list(coverage=.8, pradius=.1),
                        seqrf.args=list(k=20), dnames=names(seqdata.dom)) {

## seqdata.dom: list of stslist objects (one per channel)
## diss: list of diss matrices [[ to do:  methods to compute them.]]
## assoc: requested measure of association

## weighted does not apply when rep.method="eq.group"
## with.missing applies for rep.method="overall" and
##   with "rep" when a method is provided as diss argument,
##   i.e. when the distances must be computed.

  #assoc <- match.arg(assoc)
  test.df <- FALSE

  #if (rep.method=="eq.group") {
  #  if (!requireNamespace('TraMineRextras', quietly=TRUE))
  #        stop("This function requires the 'TraMineRextras' package.")
  #}
  if (!is.list(seqdata.dom) || length(seqdata.dom) < 2)
    stop("seqdata.dom should be a list of at least two state sequence objects")

  if (rep.method != "overall") {
    if (is.null(diss.dom) || any(is.na(diss.dom)) || !is.list(diss.dom))
        stop(" Except for rep.method='overall', diss.dom must be a list of dissimilarity matrices")
    if (is.list(diss.dom) && !(length(diss.dom) == length(seqdata.dom)))
      stop("diss.dom should be a list of same length as seqdata.dom")
    #diss.meth <- c("HAM","LCS","LCP","RLCP")
  }


  #assoclist <- c("pearson","spearman","kendall","R2","cronbach","cron.subsets","all")
  ## kendall takes too much time for large number of diss values
  assoclist <- c("LRT","V")
  assoc <- toupper(assoc)
  if (!all(assoc %in% assoclist))
    stop("bad assoc values, allowed are ", paste(assoclist, collapse=","))

  ndom <- length(seqdata.dom)
  ncases <- nrow(seqdata.dom[[1]])
  if (is.null(dnames)) dnames <- paste0('Dom',1:ndom)
  weights <- attr(seqdata.dom[[1]],"weights")
  if (is.null(weights) || !weighted) weights <- rep(1,ncases)
  if (is.null(seqrep.args[["weights"]])) seqrep.args[["weights"]] <- weights

  alph <- sapply(seqdata.dom, alphabet, simplify=FALSE)

## suppressed automatic computation of distances because of issues
#  if (is.character(diss.dom) & rep.method != "overall") {
#    meth <- diss.dom
#    diss.dom <- list()
#    for (i in 1:ndom) {
#      diss.dom[[i]] <- suppressMessages(seqdist(seqdata.dom[[i]], method=meth, with.missing=with.missing))
#    }
#  }

  repseq <- list()
  rownam <- rownames(seqdata.dom[[1]])

  if (rep.method=="rep"){
    dlist <- unique(c(names(formals(seqrep))))
    dlist <- c(dlist, "with.missing","weights")
  } else if (rep.method=="eq.group") {
    #seqrf.args[["which.plot"]] <- 'none'
    #dlist <- unique(c(names(formals(seqplot.rf))))
    dlist <- unique(c(names(formals(seqrf))))
  } else if (rep.method != "overall"){
    stop(" Unknown rep.method: ",rep.method)
  }

  for (d in 1:ndom) {
    if (rep.method=="rep"){
      seqrep.args[["seqdata"]] <- seqdata.dom[[d]]
      seqrep.args[["diss"]] <- diss.dom[[d]]
      #w.miss <- any(seqdata.dom[[d]]==attr(sedatadom[[d]],"nr"))
      if (is.null(seqrep.args[["with.missing"]])) seqrep.args[["with.missing"]] <- with.missing
      repr <- suppressMessages(do.call(seqrep, args=seqrep.args[names(seqrep.args) %in% dlist]))
      dist.rep <- attr(repr,'Distances')
      repseqid <- colnames(dist.rep)[apply(dist.rep,1,which.min)]
      repseqid <- unlist(lapply(repseqid, FUN = function(x){which(rownam==x)}))
      repseq[[d]] <- unlist(seqdata.dom[[d]][repseqid,])
    } else if (rep.method=="eq.group") {
      seqrf.args[["seqdata"]] <- seqdata.dom[[d]]
      seqrf.args[["diss"]] <- diss.dom[[d]]
      ## seqplot.rf has no weighted nor with.missing arguments.
      #repseqid <- suppressMessages(do.call(seqplot.rf, args=seqrf.args[names(seqrf.args) %in% dlist]))
      medoids <- suppressMessages(do.call(seqrf, args=seqrf.args[names(seqrf.args) %in% dlist]))
      #repseqid <- attr(medoids$rf, "kmedoid.index")
      #repseq[[d]] <- unlist(seqdata.dom[[d]][repseqid,])
      repseq[[d]] <- unlist(medoids[["seqtoplot"]])
    } else if (rep.method=="overall") {
      repseq[[d]] <- unlist(seqdata.dom[[d]])
    }
  }

  if (rep.method=="eq.group"){
    ## heights gives relative group weights summing to one
    weights <- medoids[["rf"]][["heights"]] * medoids[["rf"]][["sizes"]]["ncase"]
    #gsize <- medoids[["rf"]][["sizes"]]["gsize"] ## gsize same for all medoids, we use the last one
  }

  weights <- rep(weights,rep(ncol(seqdata.dom[[1]]),length(weights)))
  #if(rep.method=="overall") {
    lvoid <- sapply(seqdata.dom, attr, which='void')
    lnr <- NULL
    if (!with.missing) {
      lnr <- sapply(seqdata.dom, attr, which='nr')
    }
  #}

  cross.tables <- list()

  res <- matrix(NA, nrow=ndom*(ndom-1)/2, ncol=5)
  colnames(res) <- c("df","LRT","p(LRT)","v","p(v)")

  tabnames <- NULL

  k <- 0 ## res row counter
  dn.split <- strsplit(dnames," x ")
  for (d1 in 1:(ndom-1)) {
    for (d2 in (d1+1):ndom ) {
       ## cat("\n d1 = ",d1, " d2 = ", d2)
      if (!any(dn.split[[d1]] %in% dn.split[[d2]])) {
        k <- k+1

        if (is.null(wrange) | all(wrange == 0)){
          xtabl <- xtabs(weights ~ repseq[[d1]] + repseq[[d2]])
          rmiss <- which(rownames(xtabl) %in% c(lvoid[d1],lnr[d1]))
          cmiss <- which(colnames(xtabl) %in% c(lvoid[d2],lnr[d2]))
          xtabl <- xtabl[-rmiss,-cmiss] ## removing rows and columns for missing and voids
        } else {
          xtabl <- seqxtabs.win(matrix(c(repseq[[d1]],repseq[[d2]]),nrow=2,byrow=TRUE),alph1=alph[[d1]],alph2=alph[[d2]], wrange=wrange, ncases=ncases)
        }
        names(dimnames(xtabl)) <- c(dnames[d1],dnames[d2])
        tabname <- paste(dnames[d1],dnames[d2],sep="_with_")
        tabnames <- c(tabnames,tabname)
        if (cross.table==TRUE) {
          cross.tables[[tabname]] <- xtabl
        }

        ## if struct.zero set zeros as structural zeros
        start <- rep(1, length(as.vector(xtabl)))
        if (struct.zero) start[as.vector(xtabl)==0] <- 0
        nzeros <- sum(1-start)


        loglin.res <- loglin(xtabl, margin = list(1,2), start=start, print=FALSE)
        df <- loglin.res$df
        if (nzeros < df)
          df <- df - nzeros ## assuming we lose 1 df by structural zeros
        else
          test.df <- TRUE
        res[k,"df"] <- df
        if ("V" %in% assoc) {
          n <- sum(xtabl)
          nr <- nrow(xtabl)
          nc <- ncol(xtabl)
          cramerv <- chi.CramerV(loglin.res$pearson,n=n, nr=nr,nc=nc,df=df)
          res[k,"v"] <- cramerv$v
          if (p.value) {
            res[k,"p(v)"] <- cramerv$p.value
          }
        }
        if ("LRT" %in% assoc) {
          res[k,"LRT"] <- loglin.res$lrt
          if (p.value) {
            res[k,"p(LRT)"] <- pchisq(loglin.res$lrt, df=df, lower.tail = FALSE)
          }
        }
      }
    }
  }

  ## remove unused columns and rows
  res <- res[,which(!apply(res,2,function(x) {all(is.na(x))})), drop=FALSE]

  res <- res[1:k,,drop=FALSE]
  rownames(res) <- tabnames


  #res <- list()
  if (cross.table)
    attr(res, "cross.tables") <- cross.tables
  if (isTRUE(test.df))
      message("!! Some df not adjusted because more zeros than degrees of freedom")

  class(res) <- c(class(res),"sdomassoc")
  return(res)
}

chi.CramerV <- function(chisq,n,nr,nc,df){
  v <- sqrt(chisq/(n * (min(nr,nc)-1)))
  names(v) <- "Cramer V"
  p.value <- pchisq(chisq, df=df, lower.tail = FALSE)
  return(list(v=v,p.value=p.value,chisq=chisq,df=df))
}


seqxtabs.win <- function(seq2mat, alph1, alph2, wrange=c(-1L,1L), ncases=1L, weights=NULL){

  if (wrange[1] > wrange[2]) stop("wrange[1] cannot be larger than wrange[2]")

  # seq2mat matrix of sequences
  endAt <- ncol(seq2mat)/ncases
  #startAt <- 1

  wxtabl <- matrix(0, nrow=length(alph1), ncol=length(alph2))
  rownames(wxtabl) <- alph1
  colnames(wxtabl) <- alph2

  for (i in 1:length(alph1)) {
    for (j in 1:length(alph2)) {
      for (p in 1:(ncases*endAt)) {
        startseq <- ((p-1) %/% endAt)*endAt
        win <- max(startseq+1L, p+wrange[1]):min(startseq+endAt, p+wrange[2])
        for (w in win) {
          wxtabl[i,j] <- wxtabl[i,j] + (alph1[i] == seq2mat[1,p] & alph2[j] == seq2mat[2,w])
        }
      }
    }
  }
  return(wxtabl)
}


#' Generic print method for sdomassoc objects
#'
#' Prints the table of association statistics returned by \code{seqdomassoc}
#'
#' @param x \code{sdomassoc} object as returned by \code{seqdomassoc}
#' @param ... additional print arguments
#'
#' @author Gilbert Ritschard
#'
#' @export
#'
#' @exportS3Method
#' print sdomassoc
#'
#' @seealso \code{\link{seqdomassoc}}

print.sdomassoc <- function(x, ...){
  names <- dimnames(x)
  dims <- dim(x)
  attributes(x) <- NULL
  x <- as.matrix(x)
  dim(x) <- dims
  dimnames(x) <- names
  print(x, ...)
}
