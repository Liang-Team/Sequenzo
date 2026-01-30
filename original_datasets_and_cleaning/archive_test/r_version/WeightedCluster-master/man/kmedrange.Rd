\name{wcKMedRange}
\alias{wcKMedRange}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compute \code{\link{wcKMedoids}} clustering for different number of clusters.
}
\description{
Compute \code{\link{wcKMedoids}} clustering for different number of clusters.
}
\usage{
wcKMedRange(diss, kvals, weights=NULL, R=1,  samplesize=NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}}).}
  \item{kvals}{A numeric vector containing the number of cluster to compute.}
  \item{weights}{Numeric. Optional numerical vector containing case weights.}
  \item{R}{Optional number of bootstrap that can be used to build confidence intervals.}
  \item{samplesize}{Size of bootstrap sample. Default to sum of weights.}
  \item{\dots}{Additionnal parameters passed to \code{\link{wcKMedoids}}.}
}
\details{
Compute a \code{clustrange} object using the \code{\link{wcKMedoids}} method. \code{clustrange} objects contains a list of clustering solution with associated statistics and can be used to find the optimal clustering solution.

See \code{\link{as.clustrange}} for more details.
}
\seealso{
See \code{\link{as.clustrange}}.
}

\examples{
data(mvad)
## Aggregating state sequence
aggMvad <- wcAggregateCases(mvad[, 17:86], weights=mvad$weight)

## Creating state sequence object
mvad.seq <- seqdef(mvad[aggMvad$aggIndex, 17:86], weights=aggMvad$aggWeights)

## Compute distance using Hamming distance
diss <- seqdist(mvad.seq, method="HAM")

## Pam clustering
pamRange <- wcKMedRange(diss, 2:15)

## Plot all statistics (standardized)
plot(pamRange, stat="all", norm="zscoremed", lwd=3)

## Plotting sequences in 3 groups
seqdplot(mvad.seq, group=pamRange$clustering$cluster3)
}