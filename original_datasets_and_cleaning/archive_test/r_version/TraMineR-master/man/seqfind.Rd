\name{seqfind}
\alias{seqfind}
\title{Indexes of state sequence(s) x in state sequence object y}
\description{
Finds the row indexes of state sequence(s) x in the state sequence object y. 
}
\usage{
 seqfind(x, y)
}
\arguments{
  \item{x}{a state sequence object containing one or more sequences (\code{\link{seqdef}}).}
  \item{y}{a state sequence object.}
}
\seealso{
 \code{}.
}
\value{row index(es) of sequence(s) x in the set of sequences y.}

\examples{
data(mvad)
mvad.shortlab <- c("EM", "FE", "HE", "JL", "SC", "TR")
mvad.seq <- seqdef(mvad, states=mvad.shortlab, 15:86)

## Finding occurrences of sequence 176 in mvad.seq
seqfind(mvad.seq[176,],mvad.seq)

## Finding occurrences of sequence 1 to 8 in mvad.seq
seqfind(mvad.seq[1:8,],mvad.seq)
}

\author{Alexis Gabadinho (with Gilbert Ritschard for the help page)}

\keyword{Data handling}
\keyword{State sequences} 
