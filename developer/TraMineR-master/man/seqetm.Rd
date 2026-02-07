\name{seqetm}
\alias{seqetm}
\title{Create a transition-definition matrix}
\description{
This function automatically creates a transition-definition matrix from a state sequence object to  transform the state sequences into time stamped event sequences (in TSE format).
}
\usage{
seqetm(seqdata, method = "transition", use.labels = TRUE, sep = ">",
  bp = "", ep = "end", seq)
}
\arguments{
  \item{seqdata}{State sequence object from which transition events will be determined}
  \item{method}{The method to use. One of \code{"transition"}, \code{"period"} or \code{"state"}.}
  \item{use.labels}{If \code{TRUE}, transition names are built from state labels rather than from the alphabet.}
  \item{sep}{Separator to be used between the from-state and to-state that define the transition ("transition" method).}
  \item{bp}{Prefix for beginning of period event names (\code{"period"} method)}
  \item{ep}{Prefix for end of period event names (\code{"period"} method)}
  \item{seq}{Deprecated. Use \code{seqdata} instead.}
}
\details{
  Warning!!!: State labels should not contain commas \code{","} which are reserved for separating multiple events of a same transition!

  One of three methods can be selected with the \code{method} argument:

  \code{"transition"} generates a single (from-state > to-state) event for each found transition and a distinct start-state event for each different sequence start;

  \code{"period"} generates a pair of events (end-state-event, start-state-event) for each found transition, a start-state event for the beginning of the sequence and an end-state event for the end of the sequence; names used for end-state and start-state names can be controlled with the \code{bp} and \code{ep} arguments;

  \code{"state"} generates only the to-state event of each found transition (useful for analysing state sequences with methods for event sequences);
}
\value{
The transition-definition matrix.
}

\seealso{
\code{\link{seqformat}} for converting to TSE format,
\code{\link{seqecreate}} for creating an event sequence object,
\code{\link{seqdef}} for creating a state sequence object.
}
\examples{
## Creating a state sequence object from columns 13 to 24
## in the 'actcal' example data set
data(actcal)
actcal.seq <- seqdef(actcal,13:24,
        labels=c("FullTime", "PartTime", "LowPartTime", "NoWork"))
## Creating a transition matrix, one event per transition
seqetm(actcal.seq,method = "transition")

## Creating a transition matrix, single to-state events
seqetm(actcal.seq,method = "state")

## Creating a transition matrix, two events per transition
seqetm(actcal.seq,method = "period")

## changing the prefix of period start event.
seqetm(actcal.seq,method = "period", bp="begin")
}

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
\keyword{Data handling}
