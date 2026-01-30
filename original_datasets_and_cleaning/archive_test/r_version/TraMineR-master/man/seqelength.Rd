\name{seqelength}
\alias{seqelength}
\alias{seqelength<-}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Lengths of event sequences}
\description{
The length of an event sequence is its time span, i.e., the total time of observation. This information is useful to perform for instance a survival analysis. The function
\code{seqelength} retrieves the lengths of the provided sequences, while
\code{seqelength <-} sets the length of the sequences.
}
\usage{
seqelength(eseq, s)
seqelength(eseq, s) <- value
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{eseq}{An event sequence object (\code{seqelist}).}
  \item{value}{A list of sequence lengths.}
  \item{s}{Deprecated. Use \code{eseq} instead.}
}
\value{
  A numeric vector with the lengths of the sequences.
}
%\references{ ~put references to the literature/web site here ~ }
%\author{ ~~who you are~~ }
%\note{ ~~further notes~~}

%\seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{
data(actcal.tse)
actcal.eseq <- seqecreate(actcal.tse)
## Since end.event is not specified, contains no sequence lengths
## We set them manually as 12 for all sequences
sl <- numeric()
sl[1:2000] <- 12
seqelength(actcal.eseq) <- sl
actcal.eseq[1:10]
## Retrieve lengths
slen <- seqelength(actcal.eseq)
summary(slen)
}
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
\keyword{Longitudinal characteristics}
