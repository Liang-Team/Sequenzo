\name{plot.seqdiff}
\alias{plot.seqdiff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plotting a seqdiff object.
}
\description{
  Plot method for the sliding values returned by \code{\link{seqdiff}}. Plots a statistic (the Pseudo R2 by default) along the position axis.
}
\usage{
\method{plot}{seqdiff}(x, stat = "Pseudo R2", type = "l", ylab = stat,
  xlab = "", legend.pos = "top", ylim = NULL, xaxis = TRUE, col = NULL,
  xtstep = NULL, tick.last = NULL, legendposition, xaxt, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{an object produced by \code{\link{seqdiff}}}
  \item{stat}{character. Name of the statistic to be plotted. Can be any of the statistics returned by \code{\link{seqdiff}} or \code{"discrepancy"}. See details.}
  \item{type}{the line \code{type}, see \code{\link{lines}} }
  \item{ylab}{character: y-axis label.}
  \item{xlab}{character: x-axis label.}
  \item{legend.pos}{character: position of the line legend, see \code{\link{legend}}}
  \item{ylim}{numeric: if not \code{NULL}, range of the y-axis.}
  \item{xaxis}{logical: if \code{TRUE} an x-axis is plotted.}
  \item{col}{list of colors to use for each line.}
  \item{xtstep}{integer: optional step between tick-marks and labels on the x-axis. If unspecified, the \code{xtstep} attribute of the sequence object \code{x} is used. (see \code{\link{seqdef}})}
  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, the \code{tick.last} attribute of the \code{x} object is used.}
  \item{legendposition}{Deprecated. Use \code{legend.pos} instead.}
  \item{xaxt}{Deprecated. Use \code{xaxis} instead.}
  \item{\dots}{Additional parameters passed to \code{\link{lines}}}
}
\details{
  The function plots the sliding values of the requested statistic.

  You can plot the evolution of two statistics by providing for instance \code{stat=c("Pseudo R2", "Levene")}.

  Use \code{stat="discrepancy"} to plot the within-discrepancies.

  For \code{"discrepancy"}, a separate line is drawn for the whole set of sequences and for each group. Those two values cannot be paired with another statistic.
}

\seealso{\code{\link{seqdiff}}}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
\keyword{Method}
