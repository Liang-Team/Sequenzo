\name{seqecontain}
\alias{seqecontain}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Check if sequence contains events}
\description{
  Check if an event sequence or subsequence contains given events
}
\usage{
seqecontain(eseq, event.list, unknown.exclude = FALSE,
  seq, eventList, exclude)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{eseq}{A event sequence object (\code{seqelist}) or a an event subsequence object (\code{subseqelist})}
  \item{event.list}{A list of events}
  \item{unknown.exclude}{if \code{TRUE} the search is exclusive and returns \code{FALSE} for any subsequence containing an event that is not in \code{event.list}}
  \item{seq}{Deprecated. Use \code{eseq} instead.}
  \item{eventList}{Deprecated. Use \code{event.list} instead.}
  \item{exclude}{Deprecated. Use \code{unknown.exclude} instead.}
}
\details{
  Checks, for each provided event sequence, if it contains one of the events in \code{event.list}.
  If \code{unknown.exclude} is \code{TRUE}, \code{seqecontain} looks if all events of the subsequence are in \code{event.list}.
}
\value{
  A logical vector.
}
\seealso{
\code{\link{seqecreate}} for creating event sequence objects and \code{\link{seqefsub}} for creating event subsequence objects.
}
\examples{
data(actcal.tse)
actcal.eseq <- seqecreate(actcal.tse)

##Searching for frequent subsequences, that is appearing at least 20 times
fsubseq <- seqefsub(actcal.eseq,min.support=20)

##looking for subsequence with FullTime
seqecontain(fsubseq,c("FullTime"))

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
