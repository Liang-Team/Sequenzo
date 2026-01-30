\name{as.seqtree}
\alias{as.seqtree}
\alias{as.seqtree.twins}
\alias{as.seqtree.hclust}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Convert a hierarchical clustering object to a seqtree object.
}
\description{
Convert a hierarchical clustering object to a seqtree object which can then be displayed using \code{\link[TraMineR]{seqtreedisplay}}.
}
\usage{
as.seqtree(object, seqdata, diss, weighted=TRUE, ...)
\method{as.seqtree}{twins}(object, seqdata, diss, weighted=TRUE, ncluster, ...)
\method{as.seqtree}{hclust}(object, seqdata, diss, weighted=TRUE, ncluster, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{An object to be converted to a \code{\link[TraMineR]{seqtree}}.}
  \item{seqdata}{State sequence object.}
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}})}
  \item{weighted}{Logical. If \code{TRUE}, \code{weights} of the \code{seqdata} object are taken to build the tree.}
  \item{ncluster}{Maximum number of cluster. The tree will be builded until this number of cluster.}
  \item{\dots}{Additionnal parameters passed to/from methods.}
}
\details{
	By default \code{as.seqtree} try to convert the object to a \code{data.frame} assuming that it contains a list of nested clustering solutions. 
	Be aware that \code{seqtree} and \code{as.seqtree} only support binary splits.
	
	If \code{object} is an \code{hclust} or \code{twins} objects (i.e. hierarchical clustering output, see \code{\link{hclust}}, \code{\link[cluster]{diana}} or \code{\link[cluster]{agnes}}), the function returns a \code{seqtree} object reproducing the agglomerative schedulde.
}
\value{A \code{\link[TraMineR]{seqtree}} object.}

\examples{
data(mvad)
## Aggregating state sequence
aggMvad <- wcAggregateCases(mvad[, 17:86], weights=mvad$weight)

## Creating state sequence object
mvad.seq <- seqdef(mvad[aggMvad$aggIndex, 17:86], weights=aggMvad$aggWeights)

## COmpute distance using Hamming distance
diss <- seqdist(mvad.seq, method="HAM")

## Ward clustering
wardCluster <- hclust(as.dist(diss), method="ward", members=aggMvad$weight)

st <- as.seqtree(wardCluster, seqdata=mvad.seq, diss=diss, weighted=TRUE, ncluster=10)

print(st)

## You typically want to run (You need to install GraphViz before)
## seqtreedisplay(st, type="d", border=NA)

}