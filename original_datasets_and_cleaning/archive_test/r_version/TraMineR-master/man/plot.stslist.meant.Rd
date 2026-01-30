\name{plot.stslist.meant}
\alias{plot.stslist.meant}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Plot method for objects produced by the seqmeant function}
\description{
This is the plot method for objects of class \emph{stslist.meant} produced by the \code{\link{seqmeant}} function.
}
\usage{
\method{plot}{stslist.meant}(x, cpal = NULL, ylab = NULL, yaxis = TRUE,
  xaxis = TRUE, cex.axis = par("cex.axis"), ylim = NULL, bar.labels = NULL,
  cex.barlab = cex.axis, offset.barlab = .1, cex.plot, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{object of class \code{stslist.meant} as produced by the \code{seqmeant} function.}
  \item{cpal}{vector of colors of length the number of states in the alphabet. If \code{NULL} (default), the '\code{cpal}' attribute of the '\code{seqdata}' sequence object is used (see \code{\link{seqdef}}).}
  \item{ylab}{optional label of the y-axis. If set to \code{NA}, no label is drawn.}
  \item{yaxis}{should the y-axis be plotted. Default is \code{TRUE}.}
  \item{xaxis}{should the x-axis be plotted. Default is \code{TRUE}.}
  \item{cex.axis}{Tick labels magnification. See \code{\link{par}}.}
  \item{ylim}{optional vector of length 2 setting the limits of the y-axis. If \code{NULL} (default), limits are set to (0, max. sequence length).}
  \item{bar.labels}{Vector of bar labels of length equal to size of alphabet.}
  \item{cex.barlab}{Real. Bar labels magnification. Defaults to \code{cex.axis}}
  \item{offset.barlab}{Real. Vertical offset of bar labels as a proportion of \code{max(ylim)}.  Default is 0.1.}
  \item{cex.plot}{Deprecated. Use \code{cex.axis} instead.}
  \item{\dots}{further graphical parameters. For more details about the graphical parameter arguments, see \code{barplot} and \code{par}.}
}
\details{This is the plot method for the output produced by the \code{\link{seqmeant}} function, i.e., objects of class \emph{stslist.meant}. It produces a plot showing the mean times spent in each state of the alphabet.

When the \code{"se"} attribute of \code{x} is \code{TRUE}, i.e., when \code{x} contains also the standard errors of the mean times, error bars are automatically displayed on the plot. See the \code{serr} argument of  \code{\link{seqmeant}}.

The method is invoked by \code{\link{seqmtplot}} (\code{seqplot} with \code{type="mt"}), which plots mean times with automatic display of the state color legend and allows plotting by group.
}

\examples{
## Loading the mvad data set and creating a sequence object
data(mvad)
mvad.labels <- c("employment", "further education", "higher education",
                    "joblessness", "school", "training")
mvad.scodes <- c("EM","FE","HE","JL","SC","TR")
mvad.seq <- seqdef(mvad, 15:86, states=mvad.scodes, labels=mvad.labels)

## Computing the mean times
mvad.meant <- seqmeant(mvad.seq)

## Plotting
plot(mvad.meant, main="Mean durations in each state of the alphabet")

## Changing the y axis limits
plot(mvad.meant, main="Mean durations in each state of the alphabet",
 ylim=c(0,40))

## Displaying error bars
mvad.meant.e <- seqmeant(mvad.seq, serr=TRUE)
plot(mvad.meant.e, main="Mean durations in each state of the alphabet",
 ylim=c(0,40))

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Method}
