\name{seqplot.tentrop}
\alias{seqplot.tentrop}
\alias{seqplot.tentrop.m}

\title{Plotting superposed transversal-entropy curves}

\description{
Functions to plot, in a same frame, transversal-entropy curves by group or multiple curves.
 }

\usage{
seqplot.tentrop(seqdata, group, main="auto",
    col=NULL, lty=NULL, lwd=3.5, ylim=NULL, xtlab=NULL,
    xtstep=NULL, tick.last=NULL, with.legend=TRUE, glabels=NULL,
    legend.pos="topright", horiz=FALSE, cex.legend=1, ...)

seqplot.tentrop.m(seqdata.list, main="auto",
    col=NULL, lty=NULL, lwd=3.5, ylim=NULL, xtlab=NULL,
    xtstep=NULL, tick.last=NULL, with.legend=TRUE,
    glabels=names(seqdata.list),
    legend.pos="topright", horiz=FALSE, cex.legend=1, ...)
}

\arguments{
  \item{seqdata}{a state sequence object (see \code{\link[TraMineR]{seqdef}}).}

  \item{seqdata.list}{a list of state sequence objects.}

  \item{group}{a factor giving the group membership of each sequence in \code{seqdata}.}

  \item{main}{a character string giving the title of the graphic; if \code{"auto"}, a default title is printed.}

  \item{col}{a vector of colors for the different curves.}

  \item{lty}{a vector of line types for the different curves. See \code{\link{lines}}. }

  \item{lwd}{numeric or vector of numerics: width of curve lines. See \code{\link{lines}}. }

  \item{ylim}{pair of numerics defining the range for the y-axis. If left \code{NULL}, the limits are defined from the data.}

  \item{xtlab}{vector of strings defining the x-axis tick labels.}

  \item{xtstep}{integer: step between tick marks on the x-axis. If unspecified, attribute \code{xtstep} of (first) \code{seqdata} is used.}

  \item{tick.last}{logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, attribute \code{tick.last} of (first) \code{seqdata} is used.}

  \item{glabels}{a vector of strings with the curve labels. If \code{NULL} curves are labeled with the levels of the group variable or, for \code{seqplot.tentrop.m}, as \code{seq1, seq2, ...}}

  \item{with.legend}{logical: Should the legend be plotted. Default is \code{TRUE}.}

  \item{legend.pos}{legend position: default is \code{"topright"}. See\code{\link{legend} }.}

  \item{horiz}{logical: Should the legend be displayed horizontally. Set as \code{FALSE} by default, i.e., legend is displayed vertically.}

  \item{cex.legend}{Scale factor for the legend.}

  \item{...}{additional plot parameters (see \code{\link{par}}).}
}

\details{
Use \code{seqplot.tentrop} to plot curves of transversal entropies by groups of a same set of sequences, e.g. professional careers by sex.

Use \code{seqplot.tentrop.m} to plot multiple curves of transversal entropies corresponding to different sets of sequences such as sequences describing cohabitational and sequences describing occupational trajectories.
}

\value{Number \code{k} of survival curves plotted.
}

\seealso{
\code{\link[TraMineR]{seqHtplot}} for an alternative way of plotting the transversal entropies and \code{\link[TraMineR]{seqstatd} } to get the values of the entropies.
}

\examples{
## Using the biofam data which has sequences from
## ages 15 to 30 years in columns 10 to 25
data(biofam)
biofam <- biofam[1:200,]
biofam.seq <- seqdef(biofam[,10:25], xtlab=as.character(15:30), xtstep=3)

## Plotting transversal entropies by sex
seqplot.tentrop(biofam.seq, group=biofam$sex, legend.pos="bottomright")

slist <- list(woman = biofam.seq[biofam$sex=="woman",],
                man = biofam.seq[biofam$sex=="man",])
seqplot.tentrop.m(slist, legend.pos="bottomright")

## Plotting transversal entropies for women
## by father's social status
group <- biofam$cspfaj[biofam$sex=="woman"]
seqplot.tentrop(biofam.seq[biofam$sex=="woman",], group=group,
    main="Women, by father's social status", legend.pos="bottomright")

}
\keyword{Plot}
\keyword{Transversal characteristics}
