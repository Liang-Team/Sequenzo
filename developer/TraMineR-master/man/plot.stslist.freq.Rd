\name{plot.stslist.freq}
\alias{plot.stslist.freq}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Plot method for sequence frequency tables}

\description{
Plot method for output produced by the \code{seqtab} function, i.e objects of class \code{stslist.freq}.
}
\usage{
\method{plot}{stslist.freq}(x, cpal = NULL, missing.color = NULL, pbarw = TRUE,
  ylab = NULL, yaxis = TRUE, xaxis = TRUE,
  xtlab = NULL, xtstep = NULL, tick.last = NULL, cex.axis = par("cex.axis"),
  cex.plot, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{an object of class \code{stslist.freq} as produced by the \code{seqtab} function.}

  \item{cpal}{alternative color palette to be used for the states. If user specified, a vector of colors with number of elements equal to the number of states in the alphabet. By default, the \code{cpal} attribute of the \code{x} object is used.}

  \item{missing.color}{alternative color for representing missing values inside the sequences. By default, the \code{missing.color} attribute of the \code{x} object is used.}

  \item{pbarw}{if \code{pbarw=TRUE} (default), the width of the bars are proportional to the sequence frequency in the dataset.}

  \item{ylab}{label of the y axis. If set to \code{NA}, no label is drawn.}

  \item{yaxis}{if \code{TRUE} or \code{"cum"}, the y axis is plotted with a label showing the cumulated percentage frequency of the displayed sequences. If \code{"pct"}, the percentage value for each sequence is displayed.}

  \item{xaxis}{if \code{TRUE} (default) the x-axis is plotted.}

  \item{xtlab}{tick labels of the x-axis. If unspecified, the \code{names} attribute of the \code{x} object is used.}

  \item{xtstep}{interval at which the tick-marks and labels of the x-axis are displayed. For example, with xtstep=3 a tick-mark is drawn at position 1, 4, 7, etc... The display of the corresponding labels depends on the available space and is dealt with automatically. If unspecified, the \code{xtstep} attribute of the \code{x} object is used.}

  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, the \code{tick.last} attribute of the \code{x} object is used.}

  \item{cex.axis}{Axis annotation magnification. See \code{\link{par}}.}

  \item{\dots}{further graphical and \code{\link{barplot}} parameters. For example \code{border=NA} removes the bars borders, \code{space=0} removes space between sequences, \code{las} controls orientation of tick labels. See \code{\link{barplot}} and \code{\link{par}} for details. }
  \item{cex.plot}{Deprecated. Use \code{cex.axis} instead.}
}
\details{This is the plot method for output of the \code{\link{seqtab}} function, i.e., for objects of class \emph{stslist.freq}. It plots the sequences bottom-up according to their frequency in the data set.

The method is invoked by  \code{\link{seqfplot}} (\code{seqplot} with \code{type="f"}), which produces frequency plots with automatic display of the state color legend and allows plotting by group.
}
\seealso{
 \code{\link{seqfplot}}
}

\author{Alexis Gabadinho and Gilbert Ritschard}

\examples{
## Loading the 'actcal' example data set
data(actcal)

## Defining a sequence object with data in columns 13 to 24
## (activity status from january to december 2000)
actcal.lab <- c("> 37 hours", "19-36 hours", "1-18 hours", "no work")
actcal.seq <- seqdef(actcal, 13:24, labels=actcal.lab)

## 10 most frequent sequences in the data
actcal.freq <- seqtab(actcal.seq)

## Plotting the object
plot(actcal.freq, main="Sequence frequencies - actcal data set")

## Plotting all the distinct sequences without borders
## and space between sequences
actcal.freq2 <- seqtab(actcal.seq, idxs=0)
plot(actcal.freq2, main="Sequence frequencies - actcal data set",
  border=NA, space=0)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Method}
