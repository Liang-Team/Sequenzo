\name{wcClusterQuality}
\alias{wcClusterQuality}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Cluster quality statistics}
\description{
Compute several quality statistics of a given clustering solution.
}
\usage{
wcClusterQuality(diss, clustering, weights = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}})}
  \item{clustering}{Factor. A vector of clustering membership.}
  \item{weights}{optional numerical vector containing weights.}
}
\details{
Compute several quality statistics of a given clustering solution. See value for details.
}
\value{
A list with two elements \code{stats} and \code{ASW}:
	\describe{
		\item{\code{stats}}{with the following statistics:}
		\describe{
			\item{PBC}{Point Biserial Correlation. Correlation between the given distance matrice and a distance which equal to zero for individuals in the same cluster and one otherwise.}
			\item{HG}{Hubert's Gamma. Same as previous but using Kendall's Gamma coefficient.}
			\item{HGSD}{Hubert's Gamma (Somers'D). Same as previous but using Somers' D coefficient.}
			\item{ASW}{Average Silhouette width (observation).}
			\item{ASWw}{Average Silhouette width (weighted).}
			\item{CH}{Calinski-Harabasz index (Pseudo F statistics computed from distances).}
			\item{R2}{Share of the discrepancy explained by the clustering solution.}
			\item{CHsq}{Calinski-Harabasz index (Pseudo F statistics computed from \emph{squared} distances).}
			\item{R2sq}{Share of the discrepancy explained by the clustering solution (computed using \emph{squared} distances).}
			\item{HC}{Hubert's C coefficient.}
		}
		\item{\code{ASW}:}{The Average Silhouette Width of each cluster, one column for each ASW measure.}
	}
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
clust5 <- wcKMedoids(diss, k=5, weights=aggMvad$aggWeights, cluster.only=TRUE)

## Compute the silhouette of each observation
qual <- wcClusterQuality(diss, clust5, weights=aggMvad$aggWeights)

print(qual)

}
