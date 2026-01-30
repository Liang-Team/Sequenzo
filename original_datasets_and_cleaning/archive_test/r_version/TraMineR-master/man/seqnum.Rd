\name{seqnum}
\alias{seqnum}
\title{Transform into a sequence object with numerical alphabet.}

\description{
  The function \code{seqnum} transforms the provided state sequence object into an equivalent sequence object in which the original alphabet is replaced with an alphabet of numbers ranging from \code{0} to \code{(nbstates-1)}.
}
\usage{
 seqnum(seqdata, with.missing=FALSE)
}
\arguments{
	\item{seqdata}{a state sequence object as defined by the \code{\link{seqdef}} function.}

	\item{with.missing}{logical: Should missing elements in the sequences be turned into numerical values as well?  The code for missing values in the sequences is retrieved from the \code{'nr'} attribute of \code{seqdata}.}
}

\details{
The first state (for example \code{'A'}) is coded with the value \code{0}, the second state (for example \code{'B'}) is coded with the value \code{1}, etc... The function returns a sequence object containing the original sequences coded with the new numerical alphabet
ranging from \code{0} to \code{(nbstates-1)}}

\seealso{\code{\link{seqdef}}, \code{\link{alphabet} } }

\examples{
data(actcal)
actcal.seq <- seqdef(actcal,13:24)

## The first 10 sequences in the actcal.seq
## sequence object
actcal.seq[1:10,]
alphabet(actcal.seq)

## The first 10 sequences in the actcal.seq
## sequence object with numerical alphabet
seqnum(actcal.seq[1:10,])

## states A,B,C,D are now coded 0,1,2,3
alphabet(seqnum(actcal.seq))
}

\author{Alexis Gabadinho}

\keyword{Data handling}
\keyword{State sequences} 
