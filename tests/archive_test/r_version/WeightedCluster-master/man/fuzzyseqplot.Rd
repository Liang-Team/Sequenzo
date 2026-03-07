\name{fuzzyseqplot}
\alias{fuzzyseqplot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot sequences according to a fuzzy clustering.
}
\description{
This funciton propose a graphical representation of a fuzzy clustering results where sequences are weighted according to their cluster membership strength.}
\usage{
fuzzyseqplot(seqdata, group = NULL, membership.threashold = 0, type = "i", 
			members.weighted = TRUE, memb.exp = 1, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{
	State sequence object created with the \code{\link[TraMineR]{seqdef}} function.
}
  \item{group}{
	A fuzzy partition of the data, either as a membership matrix or as a \code{fanny} object.
}
  \item{membership.threashold}{
Numeric. Minimum membership strength to be included in plots.
}
  \item{type}{the type of the plot. Available types are \code{"d"} for state distribution plots (chronograms), \code{"f"} for sequence frequency plots, \code{"i"} for selected sequence index plots, \code{"I"} for whole set index plots, \code{"ms"} for plotting the sequence of modal states, \code{"mt"} for mean times plots, \code{"pc"} for parallel coordinate plots and "r" for representative sequence plots.
}
  \item{members.weighted}{
Logical. Should the sequences be weighted by their membership strength in each group before being plotted?
}
  \item{memb.exp}{
	Optional. Fuzzyness parameter used in the \code{fanny} algorithm.
}
  \item{\dots}{arguments to be passed to \code{\link[TraMineR]{seqplot}}.}
}
\details{
The dataset is augmented by repeating the sequence \eqn{s_i} of individual \eqn{i} \eqn{k} times (i.e., once per cluster). We therefore have \eqn{k} sequences for individual \eqn{i}, denoted as \eqn{s_{i1}... s_{ik}}. These sequences are therefore weighted according to their membership degree \eqn{u_{i1}... u_{ik}}. Hence, even if the same sequence were repeated \eqn{k} times, its total weight sum to 1. An additional categorical covariate is created in this augmented dataset that  specifies the cluster (ranging from 1 to \eqn{k}) of the associated membership degree. This weighting strategy allows us to use any tools available for weighted sequence data (see \code{\link[TraMineR]{seqplot}}).

For index plots, we additionally suggest ordering the sequences according to membership degree by setting \code{sortv="membership"} (see example). The most typical sequence lies at the top of the subfigures, with a high membership degree; meanwhile, the bottom shows less-characteristic patterns. Restricting to sequences with the highest membership degree can be achieved with the \code{membership.treashold} argument. 

}
\references{
Studer, M. (2018). Divisive property-based and fuzzy clustering for sequence analysis. In G. Ritschard and M. Studer (Eds.), \emph{Sequence Analysis and Related Approaches: Innovative Methods and Applications}, Life Course Research and Social Policies.
}


\seealso{
	See also \code{\link[cluster]{fanny}} for fuzzy clustering.

}
\examples{
	data(mvad)
	mvad.seq <- seqdef(mvad[1:100, 17:86])

	## COmpute distance using Hamming distance
	diss <- seqdist(mvad.seq, method="HAM")
	library(cluster)
	fclust <- fanny(diss, k=2, diss=TRUE)
	
	fuzzyseqplot(mvad.seq, group=fclust, type="d")
	fuzzyseqplot(mvad.seq, group=fclust, type="I", sortv="membership")
	fuzzyseqplot(mvad.seq, group=fclust, type="f")
}