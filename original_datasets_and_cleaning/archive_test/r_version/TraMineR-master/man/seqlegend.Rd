\name{seqlegend}
\alias{seqlegend}
\title{Plot a legend for the states in a sequence object}
\description{
Plots a legend for the states in a sequence object. Useful if several graphics are plotted together and only one legend is necessary. Unless specified by the user, the \emph{cpal} and \emph{labels} attributes of the sequence object are used for the colors and text appearing in the legend (see \code{\link{seqdef}}).
}
\usage{
seqlegend(seqdata, with.missing = "auto", cpal = NULL, missing.color = NULL,
  ltext = NULL, position = "topleft", cex = 1, boxes=TRUE, fontsize, ...)
}
\arguments{
  \item{seqdata}{a state sequence object (of class \code{stslist}) as returned by the \code{\link{seqdef}} function.}

  \item{with.missing}{logical: if set to \code{"auto"} (default), a legend for the missing state is added automatically if one or more of the sequences in seqdata contains a missing state. If \code{TRUE} a legend for the missing state is added in any case. Setting to \code{FALSE} omits the legend for the missing state.}

  \item{cpal}{alternative color palette to use for the states. If user specified, a vector of colors with number of elements equal to the number of distinct states. By default, the 'cpal' attribute of the 'seqdata' sequence object is used (see \code{\link{seqdef}}).}

	\item{missing.color}{alternative color for representing missing values inside the sequences. By default, this color is taken from the "missing.color" attribute of the sequence object being plotted.}

  \item{ltext}{optional description of the states to appear in the legend. Must be a vector of character strings with number of elements equal to the number of distinct states. If unspecified, the 'labels' attributes of the 'seqdata' sequence object is used (see \code{\link{seqdef}}). }

  \item{position}{the position of the legend in the graphic area. For accepted values, see the \code{x} argument of \code{\link{legend}}. Defaults to \code{"topleft"}.}

  \item{cex}{size of the font for the labels. A value less than 1 decreases the font size, a value greater than 1 increases the font size. Defaults to 1.}

  \item{boxes}{logical: should the colors be displayed in small square boxes? Default is \code{TRUE}. If \code{FALSE}, colors are shown using small line segments of by default length .4 and width 15. The latter can be changed by means of the \code{lwd} and \code{seg.len} \code{\link{par}} arguments.
      }
  \item{fontsize}{Deprecated. Use \code{cex} instead.}
	\item{...}{optional arguments passed to the \code{\link{legend}} function.}
}

\details{When \code{x}, \code{legend}, or \code{col} arguments are provided, they supersede respectively the \code{position}, \code{ltext}, and \code{cpal} values.
}

\value{
  Coordinate values returned by the \code{\link{legend}} function.
}
%\seealso{}
\examples{
## Loading the 'actcal' example data set
## and defining a sequence object with
## activity statuses from Jan. to Dec. 2000 (columns 13 to 24)
## of first 100 cases.
data(actcal)
actcal.seq <- seqdef(actcal[1:100,13:24],
	labels=c("> 37 hours", "19-36 hours", "1-18 hours", "no work"))

## Plotting the sequences frequency,
## the states distribution
## and the legend
par(mfrow=c(2,2))
seqIplot(actcal.seq, sortv="from.start", with.legend=FALSE)
seqfplot(actcal.seq, pbarw=TRUE, with.legend=FALSE)
seqdplot(actcal.seq, with.legend=FALSE)
seqlegend(actcal.seq)
}

\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Plot}
