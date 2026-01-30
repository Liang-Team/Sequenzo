\name{seqeweight}
\alias{seqeweight}
\alias{seqeweight<-}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	Setting or retrieving weights of an event sequence object.
}
\description{
	Event sequence objects can be weighted. Weights are used by other functions such as \code{\link{seqefsub}} or \code{\link{seqecmpgroup}} to compute weighted statistics.
}
\usage{
seqeweight(eseq, s)
seqeweight(eseq, s) <- value
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{eseq}{An event sequence object (\code{seqelist}).}
  \item{value}{Numerical vector containing weights}
  \item{s}{Deprecated. Use \code{eseq} instead.}
}
\value{
\code{seqeweight} returns a numerical vector containing the weights associated to each event sequence.
}

\examples{
##Starting with states sequences
##Loading data
data(biofam)
## Creating state sequences
biofam.seq <- seqdef(biofam,10:25,informat='STS')

## Creating event sequences from biofam
biofam.eseq <- seqecreate(biofam.seq, weighted=FALSE)

## Using the weights
seqeweight(biofam.eseq) <- biofam$wp00tbgs

## Now seqefsub accounts for weights unless weighted is set to FALSE
fsubseq <- seqefsub(biofam.eseq, pmin.support=0.01)

## Searching for weighted susbsequences which best
## discriminate the birth cohort
discr <- seqecmpgroup(fsubseq, group=biofam$birthyr>=1940)
plot(discr[1:15])

}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Sequence-object attributes}
\keyword{Event sequences}
