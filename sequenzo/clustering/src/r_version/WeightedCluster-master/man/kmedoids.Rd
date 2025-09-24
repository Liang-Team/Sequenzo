\name{wcKMedoids}
\alias{wcKMedoids}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
K-Medoids or PAM clustering of weighted data.
}
\description{
K-Medoids or PAM clustering of weighted data.
}
\usage{
	wcKMedoids(diss, k, weights=NULL, npass = 1, initialclust=NULL, 
		method="PAMonce", cluster.only = FALSE, debuglevel=0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}}).}
  \item{k}{Integer. The number of cluster.}
  \item{weights}{Numeric. Optional numerical vector containing case weights.}
  \item{npass}{Integer. Number of random start solution to test.}
  \item{initialclust}{An integer vector, a factor, an "hclust" or a "twins" object. Can be either the index of the initial medoids (length should equal to \code{k}) or a vector specifying an initial clustering solution (length should then be equal to the number of observation.). If \code{initialclust} is an "hclust" or a "twins" object, then the initial clustering solution is taken from the hierarchical clustering in \code{k} groups.}
  \item{method}{Character. One of "KMedoids", "PAM" or "PAMonce" (default). See details.}
  \item{cluster.only}{Logical. If \code{FALSE}, the quality of the retained solution is computed.}
  \item{debuglevel}{Integer. If greater than zero, print some debugging messages.}
}
\details{
	K-Medoids algorithms aim at finding the best partition of the data in a k predefined number of groups. 
	Based on a dissimilarity matrix, those algorithms seeks to minimize the (weighted) sum of distance to the medoid of each group. 
	The medoid is defined as the observation that minimize the sum of distance to the other observations of this group.
	The function \code{wcKMedoids} support three differents algorithms specified using the \code{method} argument:
	\describe{
		\item{"KMedoids"}{Start with a random solution and then iteratively adapt the medoids using an algorithm similar to kmeans. Part of the code is inspired (but completely rewritten) by the C clustering library (see de Hoon et al. 2010). If you use this solution, you should set npass>1 to try several solution.}
		\item{"PAM"}{See \code{\link[cluster]{pam}} in the \code{cluster} library. This code is based on the one available in the \code{cluster} library (Maechler et al. 2011). The advantage over the previous method is that it try to minimize a global criteria instead of a local one.}
		\item{"PAMonce"}{Same as previous but with two optimizations. First, the optimization presented by Reynolds et al. 2006. Second, only evaluate possible swap if the dissimilarity is greater than zero. This algorithm is used by default.}
	}

	wcKMedoids works differently according to the \code{diss} argument. It may be faster using a matrix but require more memory (since all distances are stored twice). 
	All combination between \code{method} and \code{diss} argument are possible, except for the "PAM" algorithm were only distance matrix may be used (use the "PAMonce" algorithm instead).
	
}

\references{

  Maechler, M., P. Rousseeuw, A. Struyf, M. Hubert and K. Hornik (2011).
  cluster: Cluster Analysis Basics and Extensions. 
  R package version 1.14.1 --- For new features, see the 'Changelog' file (in the package source).
  
  Hoon, M. d.; Imoto, S. & Miyano, S.  (2010). 
  The C Clustering Library. Manual
}
%\author{ ~~who you are~~ }

\value{An integer vector with the index of the medoids associated with each observation.}

\seealso{
\code{\link[cluster]{pam}} in the cluster library, \code{\link{wcClusterQuality}}, \code{\link{wcKMedRange}}.
}

\examples{
data(mvad)
## Aggregating state sequence
aggMvad <- wcAggregateCases(mvad[, 17:86], weights=mvad$weight)

## Creating state sequence object
mvad.seq <- seqdef(mvad[aggMvad$aggIndex, 17:86], weights=aggMvad$aggWeights)
## Computing Hamming distance between sequence
diss <- seqdist(mvad.seq, method="HAM")

## K-Medoids
clust5 <- wcKMedoids(diss, k=5, weights=aggMvad$aggWeights)

## clust5$clustering contains index number of each medoids
## Those medoids are
unique(clust5$clustering)

## Print the medoids sequences
print(mvad.seq[unique(clust5$clustering), ], informat="SPS")

## Some info about the clustering
print(clust5)

## Plot sequences according to clustering solution.
seqdplot(mvad.seq, group=clust5$clustering)

}