\name{plot.stslist.surv}
\alias{plot.stslist.surv}
\title{Plot method for objects produced by the seqsurv function}
\description{
  This is the plot method for objects of class \emph{stslist.surv} produced by
  the \code{\link{seqsurv}} function.
}
\usage{
\method{plot}{stslist.surv}(x, cpal = NULL, ylab = NULL, xlab = NULL,
  xaxis = TRUE, yaxis = TRUE, xtstep = NULL, tick.last = NULL,
  ...)
}
\arguments{
  \item{x}{
    An object of class \code{stslist.surv} as produced by the
    \code{\link{seqsurv}} function.
  }
  \item{cpal}{Vector of colors. Alternative color palette to be used for the drawn lines. The vector should be of length equal to the number of drawn survival curves, i.e., the number of selected states or number of groups when \code{x} was obtained with \code{per.state=TRUE}. When \code{cpal=NULL}, the default colors assigned to the \code{cpal} attribute of \code{x} are used.
  }
  \item{ylab}{
    Optional label for the y axis. If set as \code{NA}, no label is displayed. If \code{NULL}, a default label is used.
  }
  \item{xlab}{
    Optional label for the x axis. If set as \code{NA}, no label is displayed. If \code{NULL}, a default label is used.
  }
  \item{xaxis}{Logical. Should the x-axis be plotted. Default is \code{TRUE}.}
  \item{yaxis}{Logical. Should the y-axis be plotted. Default is \code{TRUE}.}

  \item{xtstep}{
    Optional interval at which the tick-marks of the x-axis are
    displayed. For example, with \code{xtstep = 3} a tick-mark is drawn at
    position 1, 4, 7, etc... The display of the corresponding labels depends on
    the available space and is dealt with automatically. If unspecified, the
    \code{xtstep} attribute of the \code{x} object is used.
  }
  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, the \code{tick.last} attribute of the \code{x} object is used.}

  %\item{cex.axis}{
  %  Expansion factor for the font size of the axis labels and
  %  names. The default value is 1. Values lesser than 1 will reduce the size of
  %  the font, values greater than 1 will increase the size.
  %}
  \item{\dots}{
    Further graphical parameters. For more details about the graphical parameter
    arguments, see \code{\link{plot}}, \code{\link{plot.default}} and
    \code{\link{par}}.
  }
}
\details{
  This is the plot method for the output produced by the \code{\link{seqsurv}}
  function, i.e., objects of class \emph{stslist.surv}. It displays the survival
  curves fitted for states in sequences.

%%  FIXME (Matthias).

%%  This method is called by the generic \code{\link{seqplot}} function (if
%%  \code{type = "s"}) that produces more sophisticated plots, allowing grouping
%%  and automatic display of the states legend. The \code{\link{seqsplot}}
%%  function is a shortcut for calling \code{seqplot} with \code{type = "s"}.
}
%\value{
%  FIXME (Matthias).
%}
%\references{
%  FIXME (Matthias).
%}
%\note{
%  FIXME (Matthias).
%}
\author{Matthias Studer, Gilbert Ritschard, Pierre-Alexandre Fonta}

\seealso{
  \code{\link{seqsurv}}, \code{\link{seqsplot}}, \code{\link[survival]{survfit}}.
}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam, 10:25, labels=biofam.lab)

## State survival plot
biofam.surv <- seqsurv(biofam.seq)
plot(biofam.surv)
}
\keyword{Plot}
