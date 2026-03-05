\name{wcSilhouetteObs}
\alias{wcSilhouetteObs}
\title{
Compute the silhouette of each object using weighted data.
}
\description{
	Compute the silhouette of each object using weighted data.
}
\usage{
wcSilhouetteObs(diss, clustering, weights = NULL, measure="ASW")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}})}
  \item{clustering}{Factor. A vector of clustering membership.}
  \item{weights}{optional numerical vector containing weights.}
  \item{measure}{"ASW" or "ASWw", the measure of the silhouette. See the WeigthedCluster vignettes.}
}
\details{
See the \code{\link[cluster]{silhouette}} function in the \code{cluster} package for a detailed explanation of the silhouette.
}
\value{A numeric vector containing the silhouette of each observation.}
\references{
	Maechler, M., P. Rousseeuw, A. Struyf, M. Hubert and K. Hornik (2011).
  cluster: Cluster Analysis Basics and Extensions. 
  R package version 1.14.1 --- For new features, see the 'Changelog' file (in the package source).
}

\seealso{
See also \code{\link[cluster]{silhouette}}.
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
sil <- wcSilhouetteObs(diss, clust5, weights=aggMvad$aggWeights, measure="ASWw")

## If you want to compute the average silhouette width, 
## you should take weights into account
weighted.mean(sil, w=aggMvad$aggWeights)

## Plotting sequences ordred by silhouette width, 
## best classified are draw on the top.
seqIplot(mvad.seq, group=clust5, sortv=sil)

}
