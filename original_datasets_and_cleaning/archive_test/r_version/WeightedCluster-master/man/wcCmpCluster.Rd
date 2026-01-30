\name{wcCmpCluster}
\alias{wcCmpCluster}
\alias{print.clustrangefamily}
\alias{plot.clustrangefamily}
\alias{summary.clustrangefamily}
\title{
	Automatic comparison of clustering methods.
}
\description{
	Automatically compute different clustering solutions and associated quality measures to help identifying the best one.
}
\usage{
wcCmpCluster(diss, weights = NULL, maxcluster, method = "all", pam.combine = TRUE)
\method{print}{clustrangefamily}(x, max.rank=1, ...)
\method{summary}{clustrangefamily}(object, max.rank=1, ...)
\method{plot}{clustrangefamily}(x, group="stat", method="all", pam.combine=FALSE, 
    stat="noCH", norm="none", withlegend=TRUE, lwd=1, col=NULL, legend.prop=NA, 
	rows=NA, cols=NA, main=NULL, xlab="", ylab="", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}}).}
  \item{weights}{Optional numerical vector containing weights.}
  \item{maxcluster}{Integer. Maximum number of cluster. The range will include all clustering solution starting from two to \code{ncluster}.}
  \item{method}{A vector of hierarchical clustering methods to compute or \code{"all"} for all methods. 
	Possible values include "ward", "single", "complete", "average", "mcquitty", "median", "centroid" (using \code{\link{hclust}}), 
	"pam" (using \code{\link{wcKMedRange}}), "diana" (only for unweighted datasets using \code{\link[cluster]{diana}}), 
	"beta.flexible" (only for unweighted datasets using \code{\link[cluster]{agnes}})}
  \item{pam.combine}{Logical. Should we try all combinations of hierarchical and PAM clustering?}
  \item{x}{A \code{clustrangefamily} object to plot or print}
  \item{object}{A \code{clustrangefamily} object to summarize}
  \item{max.rank}{Integer. The different number of solution to print/summarize}
  \item{group}{One of \code{"stat"} or \code{"method"}. If \code{"stat"}, plots are grouped by statistics, otherwise by clustering methods.}
  \item{stat}{Character. The list of statistics to plot or "noCH" to plot all statistics except "CH" and "CHsq" or "all" for all statistics. See \code{\link{wcClusterQuality}} for a list of possible values. It is also possible to use "RHC" to plot the quality measure 1-HC. Unlike HC, RHC should be maximized as all other quality measures.}
    \item{norm}{Character. Normalization method of the statistics can be one of "none" (no normalization), "range" (given as (value -min)/(max-min), "zscore" (adjusted by mean and standard deviation) or "zscoremed" (adjusted by median and median of the difference to the median).}
  \item{withlegend}{Logical. If \code{FALSE}, the legend is not plotted.}
  \item{lwd}{Numeric. Line width, see \code{\link{par}}.}
  \item{col}{A vector of line colors, see \code{\link{par}}. If \code{NULL}, a default set of color is used.}
  \item{legend.prop}{When \code{withlegend=TRUE}, sets the proportion of the graphic area used for plotting the legend. Default value is set according to the place (bottom or right of the graphic area) where the legend is plotted. Values from 0 to 1.}
  \item{rows,cols}{optional arguments to arrange plots.}
  \item{xlab}{x axis label.}
  \item{ylab}{y axis label.}
  \item{main}{main title of the plot.}
  \item{\dots}{Additionnal parameters passed to \code{\link{lines}}.}
}
\value{
An object of class \code{clustrangefamily} with the following elements:
	\describe{
		\item{Method name:}{the results of \code{\link{as.clustrange}} objects under each method name (see argument \code{method} for a list of possible values)}
		\item{\code{allstats}:}{A \code{matrix} containing the clustering statistics for each cluster solution and method.}
		\item{\code{param}:}{The parameters set when the function was called.}
	}
}
\seealso{
See Also \code{\link{as.clustrange}}
}	
\examples{
data(mvad)

#Creating state sequence object
mvad.seq <- seqdef(mvad[, 17:86])

# COmpute distance using Hamming distance
diss <- seqdist(mvad.seq, method="HAM")

#Ward clustering
allClust <- wcCmpCluster(diss, maxcluster=15, method=c("average", "pam", "beta.flexible"), 
                         pam.combine=FALSE)

summary(allClust, max.rank=3)

##Plot PBC, RHC and ASW
plot(allClust, stat=c("PBC", "RHC", "ASW"), norm="zscore", lwd=2)


##Plot PBC, RHC and ASW grouped by cluster method
plot(allClust, group="method", stat=c("PBC", "RHC", "ASW"), norm="zscore", lwd=2)
}
