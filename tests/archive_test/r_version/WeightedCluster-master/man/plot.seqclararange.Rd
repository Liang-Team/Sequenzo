\name{plot.seqclararange}
\alias{plot.seqclararange}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot of cluster quality of CLARA algorithm.
}
\description{
Plot of the cluster quality of a \code{\link{seqclararange} object.}
}
\usage{
\method{plot}{seqclararange}(x, stat = "CQI", type = "o", main = NULL, 
  xlab = "Number of clusters", ylab = stat, col = "blue", 
	legend.pos = "topright", pch = 19, norm = "none", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{\code{seqclararange} object, see \code{\link{seqclararange}}}
  \item{stat}{Character. The cluster quality indice to plot, namely one of \code{"CQI"} (default) to plot the value of the cluster quality indices by number of groups, \code{"stability"} to count the number of recovery of the best partition or \code{"stabmean"} to presents the average stability of the clustering.
}
  \item{type}{Character. The type of line to draw. Possible types are \code{"l"} (lines), \code{"p"} points or \code{"o"} to plot both.}
  \item{main}{Character. The  overall title of the plot: see \code{\link{title}}.}
  \item{xlab}{x axis label.}
  \item{ylab}{y axis label.}
   \item{col}{A vector of line colors, see \code{\link{par}}. If \code{NULL}, a default set of color is used.}
  \item{legend.pos}{Character. legend position, see \code{\link{legend}}.}
  \item{pch}{The plotting characters or symbols: see \code{\link{points}}.}
  \item{norm}{Character. Normalization method of the statistics can be one of "none" (no normalization), "range" (given as (value -min)/(max-min), "zscore" (adjusted by mean and standard deviation) or "zscoremed" (adjusted by median and median of the difference to the median).}
  \item{\dots}{Additionnal parameters passed to/from methods.}
}

\seealso{
See \code{\link{seqclararange}} to produce a clustering objects.
}

\references{
	Studer, M., R. Sadeghi and L. Tochon (2024). Sequence Analysis for Large Databases. \emph{LIVES Working Papers 104} \doi{10.12682/lives.2296-1658.2024.104}

}
