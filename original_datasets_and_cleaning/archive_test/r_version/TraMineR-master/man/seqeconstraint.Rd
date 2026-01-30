\name{seqeconstraint}
\alias{seqeconstraint}
\alias{print.seqeconstraint}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Setting time constraints and the counting method}
\description{
  Function used to set time constraints and the counting method in  methods (\code{seqe...}) for event sequences such as \code{\link{seqefsub}} for searching frequent subsequences or \code{\link{seqeapplysub}} for checking occurrences of subsequences.
}
\usage{
seqeconstraint(max.gap = -1, window.size = -1, age.min = -1, age.max = -1,
  age.max.end = -1, count.method = 1, maxGap, windowSize, ageMin, ageMax,
ageMaxEnd, countMethod)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{max.gap}{The maximum time gap between two events}
  \item{window.size}{The maximum time span accepted for subsequences}
  \item{age.min}{Minimal start time position allowed for subsequences. Ignored when equal to -1 (default).}
  \item{age.max}{Maximal start time position allowed for subsequences. Ignored when equal to -1 (default).}
  \item{age.max.end}{Maximal end time position allowed for subsequences. Ignored when equal to -1 (default).}
  \item{count.method}{By default, subsequences are counted only
    one time by sequence (\code{'COBJ'} method). Alternative counting methods
    are \code{'CDIST_O'}, \code{'CWIN'},
    \code{'CMINWIN'} or \code{'CDIST'} respectively. See details.}
  \item{maxGap}{Deprecated. Use \code{max.gap} instead.}
  \item{windowSize}{Deprecated. Use \code{window.size} instead.}
  \item{ageMin}{Deprecated. Use \code{age.min} instead.}
  \item{ageMax}{Deprecated. Use \code{age.max} instead.}
  \item{ageMaxEnd}{Deprecated. Use \code{age.max.end} instead.}
  \item{countMethod}{Deprecated. Use \code{count.method} instead.}
}

\details{
  \code{max.gap}, \code{window.size}, \code{age.min}, \code{age.max} and
  \code{age.max.end}. If so, two events should not be separated by more
  than \code{max.gap} and the whole subsequence should not exceed a
  \code{window.size}  time span.
  The other parameters specify the start and end age of the subsequence,
  it should start between \code{age.min} and \code{age.max} and finish
  before \code{age.max.end}.
  Parameters \code{age.min}, \code{age.max} and \code{age.max.end} are
  interpreted as the number of positions (time units) from the beginning
  of the sequence.

  There are 5 options for the \code{count.method} argument. (1) By default,
  the count is the number of sequences that contain the subsequence (\code{"COBJ"} method).
  Alternatives are (2) \code{"CDIST_O"} (counts all distinct occurrences in each sequence including possibly overlapping occurrences, i.e., occurrences sharing a same event occurrence), (3) \code{"CWIN"} (number of slidden windows of length \code{window.size}  that contain an occurrence of the subsequence),
  (4) \code{"CMINWIN"} (number of minimal windows of occurrence) and (5) \code{"CDIST"} (distinct occurrences without event occurrences overlap). See
  references.
}

\value{
  A constraint object containing one item per constraint type.
}

\references{
  Joshi, Mahesh V., George Karypis, and Vipin Kumar (2001) A Universal Formulation of
  Sequential Patterns \emph{Proceedings of the KDD'2001 Workshop on
  Temporal Data Mining}, San Francisco.

  Ritschard, G., A. Gabadinho, N.S. Müller and M. Studer (2008), Mining event sequences: A social science perspective, \emph{International Journal of Data Mining, Modelling and Management, IJDMMM}, 1(1), 68-90.
  }

\seealso{\code{\link{seqefsub}}, \code{\link{seqeapplysub}}}

\author{Matthias Studer, Nicolas S. Müller and Reto Bürgin (alternative counting methods) (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
