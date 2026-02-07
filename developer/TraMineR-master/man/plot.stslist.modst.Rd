\name{plot.stslist.modst}
\alias{plot.stslist.modst}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Plot method for modal state sequences}
\description{
Plot method for output produced by the seqmodst function, i.e objects of class stslist.modst.
}
\usage{
\method{plot}{stslist.modst}(x, cpal = NULL, ylab = NULL, yaxis = TRUE,
  xaxis = TRUE, xtlab = NULL, xtstep = NULL, tick.last = NULL,
  info = TRUE, cex.axis = par("cex.axis"), las = 1, cex.plot, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{an object of class \code{stslist.modst} as produced by the \code{seqmodst} function.}
  \item{cpal}{alternative color palette to use for the states. If user specified, a vector of colors with number of elements equal to the number of states in the alphabet. By default, the 'cpal' attribute of the \code{x} object is used.}
  \item{ylab}{an optional label for the y axis. If set to NA, no label is drawn.}

  \item{yaxis}{if TRUE (default) the y axis is plotted.}

  \item{xaxis}{if TRUE (default) the x axis is plotted.}

  \item{xtlab}{optional labels for the x axis ticks. If unspecified, the \code{names} attribute of the \code{x} object is used.}

  \item{xtstep}{optional interval at which the tick-marks and labels of the x-axis are displayed. For example, with xtstep=3 a tick-mark is drawn at position 1, 4, 7, etc... The display of the corresponding labels depends on the available space and is dealt with automatically. If unspecified, the \code{xtstep} attribute of the \code{x} object is used.}

  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, the \code{tick.last} attribute of the \code{x} object is used.}

  \item{info}{Logical: should info about frequency of occurrence of the sequence of modal states be displayed?}

  \item{cex.axis}{Axis annotation magnification. See \code{\link{par}}.}
  \item{las}{Integer in \{0, 1, 2, 3\}. Orientation of tick labels. See \code{\link{par}}.}

  \item{cex.plot}{Deprecated. Use \code{cex.axis} instead.}


  \item{\dots}{further graphical and \code{barplot} parameters. See \code{\link{barplot}} and \code{\link{par}}. }
}
\details{This is the plot method for output of \code{\link{seqmodst}}, i.e., for objects of class \emph{stslist.modst}. It plots the sequence of modal states with bar height proportional to the frequency of the modal state at each successive position.

The method is invoked by \code{\link{seqmsplot}} (\code{seqplot} with \code{type="ms"}), which in addition to the modal states  automatically displays of the state color legend and allows plotting by group.
}
\seealso{
 \code{\link{seqmsplot}}
}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam, 10:25, labels=biofam.lab)

## Modal state sequence
biofam.modst <- seqmodst(biofam.seq)
plot(biofam.modst)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Method}
