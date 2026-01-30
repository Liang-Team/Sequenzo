\name{seqdim}
\alias{seqdim}
\title{Dimension of a set of sequences}
\description{
Returns the number of sequences (rows) and the maximum length of a set of sequences.
}
\details{
The function will first search for separators '-' or ':' in the sequences in order to detect wether they are in the compressed or extended format.
}
\usage{
 seqdim(seqdata)
}
\arguments{
  \item{seqdata}{a set of sequences.}
}
\value{a vector with the number of sequences and the maximum sequence length.}
%\seealso{}
%\examples{}

\author{Alexis Gabadinho}

\keyword{Sequence-object attributes}
