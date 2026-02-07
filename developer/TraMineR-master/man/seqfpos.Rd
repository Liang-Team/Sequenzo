\name{seqfpos}
\alias{seqfpos}
\title{Search for the first occurrence of a given element in a sequence}
\description{
Returns a vector containing the position of the first occurrence of the given element in each of the sequences in the data set.
}
\details{the state to search for has to be passed as a character string, and must be one of the state returned by the \code{\link{alphabet}} function. If the state is not contained in a sequence, NA is returned for this sequence.}
\usage{
 seqfpos(seqdata, state)
}
\arguments{
  \item{seqdata}{a sequence object (see \code{\link{seqdef}} function).}
  \item{state}{the state element to search in the sequences}
}

\examples{
data(biofam)
biofam.seq <- seqdef(biofam,10:25)

## Searching for the first occurrence of state 1
## in each of the 5 first sequence of the biofam data set.
seqfpos(biofam.seq[1:5,],"1")
}

\author{Alexis Gabadinho}

\keyword{Longitudinal characteristics}
