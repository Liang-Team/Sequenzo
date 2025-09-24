\name{bootclustrange}
\alias{bootclustrange}
\alias{print.bootclustrange}
\alias{plot.bootclustrange}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Cluster Quality Indices estimation by subsampling}
\description{
\code{bootclustrange} estimates the quality of the clustering based on subsamples of the data to avoid computational overload.
}
\usage{
bootclustrange(object, seqdata, seqdist.args = list(method = "LCS"),
               R = 100, sample.size = 1000, parallel = FALSE,
               progressbar = FALSE, sampling = "clustering",
               strata = NULL)
\method{plot}{bootclustrange}(x, stat = "noCH", legendpos = "bottomright",
                              norm = "none", withlegend = TRUE, lwd = 1,
                              col = NULL, ylab = "Indicators", 
                              xlab = "N clusters", conf.int = 0.95, 
                              ci.method = "perc", ci.alpha = 0.3, 
                              line = "median", ...)
\method{print}{bootclustrange}(x, digits = 2, bootstat = c("mean"), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{A \code{\link{seqclararange}} \code{object} or a \code{data.frame} with the clustering to be evaluated.}
\item{seqdata}{State sequence object of class \code{stslist}. The sequence data to use. Use \code{\link[TraMineR]{seqdef}} to create such an object.}
\item{seqdist.args}{List of arguments passed to \code{\link[TraMineR]{seqdist}} for computing the distances.}
\item{R}{Numeric. The number of subsamples to use.}
 \item{sample.size}{Numeric. The size of the subsamples, values between 1000 and 10 000 are recommended.}
  \item{parallel}{Logical. Whether to initialize the parallel processing of the \code{future} package using the default \code{\link[future]{multisession}} strategy. If \code{FALSE} (default), then the current \code{\link[future]{plan}} is used. If \code{TRUE}, \code{\link[future]{multisession}} \code{\link[future]{plan}} is initialized using default values.}
	\item{progressbar}{Logical. Whether to initialize a progressbar using the \code{future} package. If \code{FALSE} (default), then the current progress bar \code{\link[progressr]{handlers}} is used . If \code{TRUE}, a new global progress bar \code{\link[progressr]{handlers}} is initialized.}
  
  \item{sampling}{Character. The sampling procedure to be used: \code{"clustering"} (default) the sampling is stratified by the maximum number of clusters, use \code{"medoids"} to add the medoids in each subsamples, \code{"strata"} to stratify by the \code{strata} arguments, or \code{"random"} for random sampling.
}
  \item{strata}{An optional stratification variable.}
  \item{x}{A \code{bootclustrange} object to be plotted or printed.}
  \item{stat}{Character. The list of statistics to plot or "noCH" to plot all statistics except "CH" and "CHsq" or "all" for all statistics. See \code{\link{as.clustrange}} for a list of possible values.}
  \item{legendpos}{Character. legend position, see \code{\link{legend}}.}
  \item{norm}{Character. Normalization method of the statistics can be one of "none" (no normalization), "range" (given as (value -min)/(max-min), "zscore" (adjusted by mean and standard deviation) or "zscoremed" (adjusted by median and median of the difference to the median).}
  \item{withlegend}{Logical. If \code{FALSE}, the legend is not plotted.}
  \item{lwd}{Numeric. Line width, see \code{\link{par}}.}
  \item{col}{A vector of line colors, see \code{\link{par}}. If \code{NULL}, a default set of color is used.}
  \item{xlab}{x axis label.}
  \item{ylab}{y axis label.}
  \item{conf.int}{Confidence to build the confidence interval (default: 0.95).}
  \item{ci.method}{Method used to build the confidence interval (only if bootstrap has been used, see R above). One of "none" (do not plot confidence interval), "norm" (based on normal approximation), "perc" (default, based on percentile).)}
  \item{ci.alpha}{alpha color value used to plot the interval.}
  \item{line}{Which value should be plotted by the line? One of "mean" (average over all bootstraps), "median"(default, median over all bootstraps).}
  \item{digits}{Number of digits to be printed.}
  \item{bootstat}{The summary statistic to use \code{"mean"} or \code{"median"}.}
  \item{\dots}{Additionnal parameters passed to/from methods.}
}
\details{
\code{bootclustrange} estimates the quality of the clustering based on subsamples of the data to avoid computational overload. It randomly samples \code{R} times \code{sample.size} sequences from \code{seqdata} using the sampling procedure defined by the \code{sampling} arguments. In each subsample, a distance matrix is computed using the selected sequences and the \code{seqdist.args} arguments and the cluster quality indices are then estimated using \code{\link{as.clustrange}}.

The clustering can be specified either as a \code{\link{seqclararange}} object or a \code{data.frame}.
}
\value{
A \code{clustrange} object, see \code{\link{as.clustrange}} with the bootrapped values.
}
\references{
	Studer, M., R. Sadeghi and L. Tochon (2024). Sequence Analysis for Large Databases. \emph{LIVES Working Papers 104} \doi{10.12682/lives.2296-1658.2024.104}
}


\seealso{
See Also \code{\link{as.clustrange}} for the list of cluster quality indices that are computed, and \code{\link{seqclararange}} for example of use
}

