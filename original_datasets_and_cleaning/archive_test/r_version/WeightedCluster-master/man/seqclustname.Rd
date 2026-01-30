\name{seqclustname}
\alias{seqclustname}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Automatic labeling of cluster using sequence medoids
}
\description{
This function automatically name the cluster using the sequence medoid of each cluster.
}
\usage{
seqclustname(seqdata, group, diss, weighted = TRUE, perc = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{State sequence object (see \code{\link[TraMineR]{seqdef}}).}
  \item{group}{A vector of clustering membership.}
  \item{diss}{a dissimilarity matrix or a \code{dist} object.}
  \item{weighted}{Logical. If \code{TRUE}, \code{weights} of the \code{seqdata} object are taken to find the medoids.}
  \item{perc}{Logical. If \code{TRUE}, the percentage of sequences in each cluster is added to the label of each group.}
}
\value{A factor of clustering membership. The labels are defined using sequences medoids and optionnaly percentage of case in each cluster.
}

\examples{
data(mvad)
## Aggregating state sequence
aggMvad <- wcAggregateCases(mvad[, 17:86], weights=mvad$weight)

## Creating state sequence object
mvad.seq <- seqdef(mvad[aggMvad$aggIndex, 17:86], weights=aggMvad$aggWeights)
## Computing Hamming distance between sequence
diss <- seqdist(mvad.seq, method="HAM")

## KMedoids using PAMonce method (clustering only)
clust5 <- wcKMedoids(diss, k=5, weights=aggMvad$aggWeights)

clust5.labels <- seqclustname(mvad.seq, clust5$clustering, diss=diss, perc=TRUE)
seqdplot(mvad.seq, group=clust5.labels)
}
