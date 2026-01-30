\name{seqpropclust}
\alias{seqpropclust}
\alias{wcPropertyClustering}
\alias{dtcut}

%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Monothetic clustering of state sequences
}
\description{
Monothetic divisive clustering of the data using object properties. For state sequences object different set of properties are automoatically extracted.
}
\usage{
seqpropclust(seqdata, diss, properties = c("state", "duration", "spell.age", 
		"spell.dur", "transition", "pattern", "AFtransition", "AFpattern", 
		"Complexity"), other.prop = NULL, prop.only = FALSE, pmin.support = 0.05, 
		max.k = -1, with.missing = TRUE, R = 1, weight.permutation = "diss", 
		min.size = 0.01, max.depth = 5, maxcluster = NULL, ...)
		
wcPropertyClustering(diss, properties, maxcluster = NULL, ...)
dtcut(st, k, labels = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
 \item{seqdata}{State sequence object (see \code{\link[TraMineR]{seqdef}}).}
  \item{diss}{a dissimilarity matrix or a \code{dist} object.}
  \item{properties}{Character or \code{data.frame}. In \code{seqpropclust}, it can be a list of properties to be extracted from \code{seqdata}. It can also be a \code{data.frame} specifying the properties to use for the clustering.}
  \item{other.prop}{\code{data.frame}. Additional properties to be considered to cluster the sequences.}
  \item{prop.only}{Logical. If \code{TRUE}, the function returns a data.frame containing the extracted properties (without clustering the data).}
  \item{pmin.support}{Numeric. Minimum support (as a proportion of sequences). See \code{\link[TraMineR]{seqefsub}}.}
  \item{max.k}{Numeric. The maximum number of events allowed in a subsequence. See \code{\link[TraMineR]{seqefsub}}.}
  \item{with.missing}{Logical. If \code{TRUE}, property of missing spell are also extracted.}
  \item{R}{Number of permutations used to assess the significance of the split. See \code{\link[TraMineR]{disstree}}.}
  \item{weight.permutation}{Weight permutation method: "diss" (attach weights to the dissimilarity matrix), "replicate" (replicate cases using weights), "rounded-replicate" (replicate case using rounded weights), "random-sampling" (random assignment of covariate profiles to the objects using distributions defined by the weights.). See \code{\link[TraMineR]{disstree}}.}
  \item{min.size}{Minimum number of cases in a node, will be treated as a proportion if less than 1. See \code{\link[TraMineR]{disstree}}.}
  \item{max.depth}{Maximum depth of the tree. See \code{\link[TraMineR]{disstree}}.}
  \item{maxcluster}{Maximum number of cluster to consider.}
 \item{st}{A divise clustering tree as produced by \code{seqpropclust}}
  \item{k}{The number of groups to extract.}
  \item{labels}{Logical. If \code{TRUE}, rules to assign an object to a sequence is used to label the cluster (instead of a number).}
  \item{\dots}{Arguments passed to/from other methods.}
}
\details{
The method implement the DIVCLUS-T algorithm.

}
\value{
Return a \code{seqpropclust} object, which is (in fact) a \code{distree} object.  See \code{\link[TraMineR]{disstree}}.
}
\references{

Studer, M. (2018). Divisive property-based and fuzzy clustering for sequence analysis. In G. Ritschard and M. Studer (Eds.), \emph{Sequence Analysis and Related Approaches: Innovative Methods and Applications}, Life Course Research and Social Policies. Springer.

Piccarreta R, Billari FC (2007). Clustering work and family trajectories by using a divisive algorithm. \emph{Journal of the Royal Statistical Society: Series A (Statistics in Society)}, 170(4), 1061-1078.

Chavent M, Lechevallier Y, Briant O (2007). DIVCLUS-T: A monothetic divisive hierarchical clustering method. \emph{Computational Statistics & Data Analysis}, 52(2), 687-701.
}
%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{as.clustrange}}, \code{\link[TraMineR]{seqtreedisplay}}, \code{\link[TraMineR]{disstree}}.
}


\examples{
data(mvad)
mvad.seq <- seqdef(mvad[1:100, 17:86])

## COmpute distance using Hamming distance
diss <- seqdist(mvad.seq, method="HAM")

pclust <- seqpropclust(mvad.seq , diss=diss, maxcluster=5, properties=c("state", "duration")) 

## Run it to visualize the results
##seqtreedisplay(pclust, type="d", border=NA, showdepth=TRUE)

pclustqual <- as.clustrange(pclust, diss=diss, ncluster=5)
}
