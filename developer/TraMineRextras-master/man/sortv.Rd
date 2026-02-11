\encoding{UTF-8}
\name{sortv}
\alias{sortv}
\alias{sorti}

\author{Gilbert Ritschard}

\title{Sort sequences by states at the successive positions}


\description{
Returns a sorting vector to sort state sequences in a TraMineR sequence object (\code{\link[TraMineR]{seqdef}}) by the states at the successive positions.}

\usage{
sorti(seqdata, start = "end", sort.index=TRUE)

sortv(seqdata, start = "end")

}

\arguments{
  \item{seqdata}{A state sequence object as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{start}{Where to start the sort. One of \code{"beg"} (beginning) or \code{"end"}.}

  \item{sort.index}{Should the function return sort indexes? If \code{FALSE}, sort values are returned.}
}

\details{
    With \code{start = "end"} (default), the primary sort key is the final state, then the previous one and so on. With \code{start = "beg"}, the primary sort key is the state at the first position, then at the next one and so on.

    With \code{sort.index = FALSE}, the function returns a vector of values whose order will determine the wanted order. This should be used as \code{sortv} argument of the \code{\link[TraMineR]{seqiplot}} function. With \code{sort.index = TRUE}, the function returns a vector of indexes to be used for indexing.

    The \code{sortv} form is an alias for \code{sorti(..., sort.index = FALSE)}.
}

\value{
  If \code{sort.index = FALSE}, the vector of sorting values.\cr
  Otherwise the vector of sorting indexes.
}

\seealso{
Details about \code{type = "i"} or \code{type = "I"} in
\code{\link[TraMineR]{seqplot}} .
}

\examples{
data(actcal)
actcal.seq <- seqdef(actcal[1:100,13:24])
par(mfrow=c(1,2))
seqIplot(actcal.seq, sortv=sortv(actcal.seq), with.legend = FALSE)
seqIplot(actcal.seq, sortv=sortv(actcal.seq, start="beg"), with.legend = FALSE)
actcal.seq[sorti(actcal.seq)[90:100],]

data(mvad)
mvad.seq <- seqdef(mvad[1:100,17:86])
par(mfrow=c(1,2))
seqIplot(mvad.seq, sortv=sortv(mvad.seq, start="end"), with.legend = FALSE)
seqIplot(mvad.seq, sortv=sortv(mvad.seq, start="beg"), with.legend = FALSE)
print( mvad.seq[sorti(mvad.seq,start="beg")[90:100],], format="SPS")

}
\keyword{util}
