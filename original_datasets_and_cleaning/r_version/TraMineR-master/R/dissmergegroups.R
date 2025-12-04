#' Merging groups by minimizing loss of partition quality.
#'
#'
#'
#' @details
#'
#' The procedure is greedy. The function iteratively searches for the pair of groups whose merge minimizes quality loss. As long as the smallest group is smaller than \code{small}, it searches among the pairs formed by that group with one of the other groups. Once all groups have sizes larger than \code{small}, the search is done among all possible pairs of groups. There are two stopping criteria: the minimum number of groups (\code{min.group}) and maximum allowed quality deterioration (\code{crit}). The percentage specified with \code{crit} applies either to the quality of the initial partition (\code{ref="initial"}), the quality after the previous iteration (\code{ref="previous"}), or the maximal quality achieved so far (\code{ref="max"}), the latter being the default. The process stops when any of the criteria is reached.
#'
#'
#' @param diss A dissimilarity matrix or a distance object.
#' @param group Group membership. Typically, the outcome of a clustering function.
#' @param weights Vector of non-negative case weights.
#' @param measure Character. Name of quality index. One of those returned by \code{\link[WeightedCluster]{wcClusterQuality}}
#' @param crit Positive real value in the range [0,1]. Maximal allowed proportion of quality loss.
#' @param ref Character. Reference for proportion \code{crit}. One of \code{"initial"}, \code{"max"} (default), and \code{"previous"}.
#' @param min.group Integer. Minimal number of end groups.
#' @param small real. Percentage of sample size under which groups are considered as small.
#' @param silent Logical. Should merge steps be displayed during computation?
#'
#' @return Vector of merged group memberships.
#'
#' @seealso \code{\link[WeightedCluster]{wcClusterQuality}}
#'
#' @references
#' Ritschard, G., T.F. Liao, and E. Struffolino (2023). Strategies for
#' multidomain sequence analysis in social research.
#' *Sociological Methodology*, in press.
#'
#' @author Gilbert Ritschard
#'
#' @export
#'
#' @importFrom WeightedCluster wcClusterQuality
#' @importFrom stats xtabs
#'
#' @examples
#' data(biofam)
#'
#' ## Building one channel per type of event (children, married, left home)
#' cases <- 1:50
#' bf <- as.matrix(biofam[cases, 10:25])
#' children <-  bf==4 | bf==5 | bf==6
#' married <- bf == 2 | bf== 3 | bf==6
#' left <- bf==1 | bf==3 | bf==5 | bf==6
#'
#' ## Creating sequence objects
#' child.seq <- seqdef(children, weights = biofam[cases,'wp00tbgs'])
#' marr.seq <- seqdef(married, weights = biofam[cases,'wp00tbgs'])
#' left.seq <- seqdef(left, weights = biofam[cases,'wp00tbgs'])
#'
#' ## distances by channel
#' dchild <- seqdist(child.seq, method="OM", sm="INDELSLOG")
#' dmarr <- seqdist(marr.seq, method="OM", sm="INDELSLOG")
#' dleft <- seqdist(left.seq, method="OM", sm="INDELSLOG")
#' dnames <- c("child","marr","left")
#'
#' if (require("WeightedCluster")){
#'    child.cl2 <- factor(wcKMedoids(dchild, k=2)[["clustering"]])
#'    marr.cl2 <- factor(wcKMedoids(dmarr, k=2)[["clustering"]])
#'    left.cl2 <- factor(wcKMedoids(dleft, k=2)[["clustering"]])
#'
#'    ## Multidomain sequences
#'    MD.seq <- seqMD(list(child.seq,marr.seq,left.seq))
#'    d.expand <- seqdist(MD.seq, method="LCS")
#'    clust.comb <- interaction(child.cl2,marr.cl2,left.cl2)
#'    merged.grp <- dissmergegroups(d.expand, clust.comb,
#'                          weights=biofam[cases,'wp00tbgs'])
#'
#'    ## weighted size of merged groups
#'    xtabs(biofam[cases,'wp00tbgs'] ~ merged.grp)
#' } else cat("You must install WeightedCluster to run this example")


dissmergegroups <- function(diss, group, weights=NULL, measure="ASW",
                            crit=.2, ref="max", min.group=4,
                            small=.05,
                            silent=FALSE){

  if (!(is.matrix(diss) && nrow(diss)==ncol(diss)) && !inherits(diss,"dist"))
    msg.stop("diss must be a matrix of pairwise distances or a distance object")

  if (any(is.na(group)) || is.null(group))
    msg.stop("group must be a vector or factor with at least min.group different values!")

  if (inherits(diss,"dist"))
    dsize <- attr(diss,"Size")
  else
    dsize <- nrow(diss)

  if (length(group)!=dsize)
    msg.stop("length of group non conformable with diss")

  ## group sizes
  if (is.null(weights)) {
    N <- nrow(diss)
    w <- rep(1,N)
  } else {
    w <- weights
    N <- sum(weights)
  }



  ## factor suppresses empty groups
  gn <- as.integer(factor(group))

  #if(!silent) cat("Original group numbers: ",levels(as.factor(gn)),"\n")
  maxgn <- max(gn)
  empty.groups <- which(xtabs(w~group)==0)
  if (length(empty.groups)>0){
    if(!silent) {
      cat("Suppressing empty groups: ",empty.groups,"\n",
                    "remaining groups renumbered from 1 to ", maxgn,"\n")
      cat("Measure=",measure,", crit=",crit,", ref=",ref,", min.group=", min.group, "\n")
    }
  }


  if (small < 1)
    minsize <- small*N
  else
    minsize <- small

  finalqual <- quality <- quality.ref <- WeightedCluster::wcClusterQuality(diss, gn, weights)[["stats"]][measure]

  #  k<-0
  while(maxgn > min.group) {
    # k <- k+1
    # print(c(maxgn,k))
    diff <- quality.ref
    merge.ij <- NULL
    grp.sizes <- xtabs(w ~ gn)
    if (min(grp.sizes) > minsize){
      for (i in 1:(maxgn-1)){
        for (j in (i+1):maxgn){
          #print(c(i,j))
          gng <- gn
          gng[gn==j] <- i
          qual <- WeightedCluster::wcClusterQuality(diss, gng, weights)[["stats"]][measure]
          if(quality.ref - qual < diff){
            merge.ij <- c(i,j)
            diff <- quality.ref - qual
          }
        }
      }
    }
    else {
      i <- which.min(grp.sizes)
      for (j in 1:maxgn){
        if (j != i){
          #print(c(i,j))
          gng <- gn
          gng[gn==j] <- i
          qual <- WeightedCluster::wcClusterQuality(diss, gng, weights)[["stats"]][measure]
          if(quality.ref - qual < diff){
            if (i<j)
              merge.ij <- c(i,j)
            else
              merge.ij <- c(j,i)
            diff <- quality.ref - qual
          }
        }
      }
    }
    if (diff > crit*quality.ref )
      break
    else {#merge and continue
      if (!silent) cat("Merging groups ",merge.ij[1]," and ",merge.ij[2],"\n")
      gn[gn==merge.ij[2]] <- merge.ij[1]
      gn <- as.integer(as.factor(gn))
      maxgn <- max(gn)
      finalqual <- quality.ref - diff
      if (ref=="max"){
        quality.ref <- max(quality.ref, finalqual)
      }
      else if (ref=="previous") {
        quality.ref <- finalqual
      }
      else if (ref=="initial") {
        quality.ref <- quality
      }
      else stop("Unknown ref value!")
    }

  }

  names(finalqual) <- measure
  attr(gn,"qual") <- finalqual

  return(gn)
}
