\name{seqsep}
\alias{seqsep}
\title{Adds separators to sequences stored as character string}
\description{
Adds separators to sequences stored as character string.
}
\usage{
 seqsep(seqdata, sl=1, sep="-")
}
\arguments{
  \item{seqdata}{a dataframe or matrix containing sequence data, as vectors of states or events.}
  \item{sl}{the length of the states (the number of characters used to represent them). Default is 1.}
  \item{sep}{the character used as separator. Set by default as \code{"-"}.}
}
\seealso{
 \code{\link{seqdecomp}}.
}
\examples{
seqsep("ABAAAAAAD")}
\keyword{Data handling}
