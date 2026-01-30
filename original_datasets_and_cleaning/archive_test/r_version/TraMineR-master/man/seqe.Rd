\name{is.eseq}
\alias{is.eseq}
\alias{is.seqe}
\alias{is.seqelist}
\alias{print.eseq}
\alias{print.seqelist}
\alias{levels.seqelist}
\alias{[.seqelist}
\alias{levels.eseq}
\alias{Math.eseq}
\alias{Math.seqelist}
\alias{Ops.eseq}
\alias{Ops.seqelist}
\alias{Summary.eseq}
\alias{Summary.seqelist}


\title{Event sequence object}
\description{
  TraMineR uses an internal event sequence object for all its operations
  with event sequences such as mining frequent subsequences. The
  function \code{is.seqelist} checks wether the argument is an event
  sequence object while \code{is.eseq} checks wether the argument is a
  single element of an event sequence object.
  %\code{as.seqelist} transform a list of event sequences into an internal event sequence object.

  There is a print method that can be applied to such event sequence object.

  is.seqe() has been removed in TraMineR 2.x, use is.eseq() instead.
}
\usage{
is.eseq(eseq, s)
is.seqelist(eseq, s)
}
\arguments{
  \item{eseq}{Sequence}
  \item{s}{Deprecated. Use \code{eseq} instead.}
}
\seealso{\code{\link{str.seqelist}}, \code{\link{str.eseq}},\code{\link{as.character.seqelist}} and \code{\link{as.character.eseq}} to convert sequences from an event sequence object into character strings}
%\examples{}

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
\keyword{internal}
