\name{seqtabstocc}
\alias{seqtabstocc}
%
\author{Gilbert Ritschard}
%
\title{Frequencies of state co-occurrence patterns}
%
\description{
  Computes the frequencies of co-occurring state patterns.
}
\usage{
  seqtabstocc(seqdata, with.missing=FALSE, ...)
}
\arguments{
  \item{seqdata}{ A state sequence (\code{stslist}) object as returned by \code{\link[TraMineR]{seqdef}}.}
  \item{with.missing}{Logical. Should the missing state be considered as a regular state?}
  \item{...}{ Additional arguments to be passed to \code{\link[TraMineR]{seqtab}}.}
}
\details{
  The function extracts the list of states co-occurring in each sequence. For each sequence, the co-occurring states are extracted as the sequence of the alphabetically sorted distinct states. The frequencies of the extracted sets of states is then obtained by means of the  TraMineR \code{\link[TraMineR]{seqtab}} function.

  Returned patterns with a single state correspond to sequences that contain only that state.
}
\value{
  A \code{stslist.freq} object with co-occurrence patterns sorted in descending frequency order.
}
\seealso{
\code{\link[TraMineR]{seqtab}}
}
\examples{
## Creating a sequence object from the first 500 actcal data.
data(actcal)
actcal.seq <- seqdef(actcal[1:500,13:24])

## 10 most frequent state patterns in the data
seqtabstocc(actcal.seq)

## All state patterns
seqtabstocc(actcal.seq, idxs=0)

## Example with missing states
data(ex1)
ex1 <- ex1[,1:13] ## dropping last weight column
## adding 3 sequences with no gap and left missing state
ex1 <- rbind(ex1,c(rep("A",4),rep(NA,9)))
ex1 <- rbind(ex1,c(rep("A",4),rep(NA,9)))
ex1 <- rbind(ex1,rep("A",13))
s.ex1 <- seqdef(ex1)
seqtabstocc(s.ex1, with.missing=TRUE)

}
\keyword{util}
