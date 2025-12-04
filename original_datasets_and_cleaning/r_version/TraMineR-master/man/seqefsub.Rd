\name{seqefsub}
\alias{seqefsub}
%\alias{[.subseqelist}
%\alias{createsubseqelist}
%\alias{seqecreatesub}
\alias{is.subseqelist}
\alias{print.subseqelist}
\title{Searching for frequent subsequences}
\description{
  Returns the list of subsequences with minimal support sorted in decreasing order of support. Various time constraints can be set to restrict the search to specific time periods or subsequence durations. The function permits also to get information on specified subsequences.
}
\details{
  There are two usages of this function. The first is for searching subsequences satisfying a support condition.
  By default, the support is counted per sequence and not per occurrence, i.e. when a sequence contains several occurrences of a same subsequence it is counted only once. Use the \code{count.method} argument of \code{\link{seqeconstraint}} to change that. The minimal required support can be set with \code{pmin.support} as a proportion (between 0 and 1) in which case the support will be rounded, or through min.support as a number of sequences.
  Time constraints can also be imposed with the \code{constraint} argument, which must be the outcome of a call to the \code{\link{seqeconstraint}} function.

  The second possibility is for searching sequences that contain specified subsequences. This is done by passing the list of subsequences with the \code{str.subseq} argument. The subsequences must contain only events from the alphabet of events of \code{eseq} and must be in the same format as that used to display subsequences (see \code{\link{str.seqelist}}).
  Each transition (group of events) should be enclosed in parentheses () and separated with commas, and the succession of transitions should be denoted by a '-' indicating a time gap.
  For instance "(FullTime)-(PartTime, Children)" stands for the subsequence "FullTime" followed by the transition defined by the two simultaneously occurring events "PartTime" and "Children".

  To get information such as the number of occurrences of the subsequences returned by \code{seqefsub} or the sequences that contain each subsequence use the function \code{\link{seqeapplysub}}.

  Subsets of the returned \code{subseqelist} can be accessed with the \code{[]} operator (see example). There are print and plot methods for \code{subseqelist}.
}
\usage{
seqefsub(eseq, str.subseq = NULL, min.support = NULL,
  pmin.support = NULL, constraint = seqeconstraint(), max.k = -1,
  weighted = TRUE, seq, strsubseq, minSupport, pMinSupport, maxK)
}

\arguments{
  \item{eseq}{A list of event sequences}
  \item{str.subseq}{A list of specific subsequences to look for. See details.}
  \item{min.support}{The minimum support (in number of sequences)}
  \item{pmin.support}{The minimum support (in percentage, corresponding count will be rounded)}
  \item{constraint}{A time constraint object as returned by \code{\link{seqeconstraint}}}
  \item{max.k}{The maximum number of events allowed in a subsequence}
  \item{weighted}{Logical. Should \code{seqefsub} use the weights specified in \code{eseq} (see \code{\link{seqeweight}}).}
  \item{seq}{Deprecated. Use \code{eseq} instead.}
  \item{strsubseq}{Deprecated. Use \code{str.subseq} instead.}
  \item{minSupport}{Deprecated. Use \code{min.support} instead.}
  \item{pMinSupport}{Deprecated. Use \code{pmin.support} instead.}
  \item{maxK}{Deprecated. Use \code{max.k} instead.}
}
\value{
A \code{subseqelist} object with at least the following attributes:
  \item{eseq}{The list of sequences in which the subsequences were searched (a \code{seqelist} event sequence object).}
  \item{subseq}{A list of subsequences (a \code{seqelist} event sequence object).}
  \item{data}{A data frame containing details (support, frequency, ...) about the subsequences}
  \item{constraint}{The constraint object used when searching the subsequences.}
  \item{type}{The type of search: 'frequent' or 'user'}
  %\item{count}{The number of sequences.}
}
\seealso{
 See \code{\link{plot.subseqelist}} to plot the result.
 See \code{\link{seqecreate}} for creating event sequences. 
 See \code{\link{seqeapplysub}} to count the number of occurrences of frequent subsequences in each sequence.
 See \code{\link{is.seqelist}} about \code{seqelist}.
}

\references{
  Ritschard, G., Bürgin, R., and Studer, M. (2014), "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) \emph{Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences}. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.
}


\examples{
data(actcal.tse)
actcal.eseq <- seqecreate(actcal.tse)

## Searching for subsequences appearing at least 20 times
fsubseq <- seqefsub(actcal.eseq, min.support=20)
## The same using a percentage
fsubseq <- seqefsub(actcal.eseq, pmin.support=0.01)
## Getting a string representation of subsequences
## First ten most frequent subsequences
fsubseq[1:10]

## Using time constraints
## Looking for subsequences starting in Summer (between June and September)
fsubseq <- seqefsub(actcal.eseq, min.support=10,
  constraint=seqeconstraint(age.min=6, age.max=9))
fsubseq[1:10]

##Looking for subsequences occurring in Summer (between June and September)
fsubseq <- seqefsub(actcal.eseq, min.support = 10,
  constraint=seqeconstraint(age.min=6, age.max=9, age.max.end=9))
fsubseq[1:10]

##Looking for subsequence enclosed in a 6 month period
## and with a maximum gap of 2 month
fsubseq <- seqefsub(actcal.eseq, min.support=10,
  constraint=seqeconstraint(max.gap=2, window.size=6))
fsubseq[1:10]
}

\author{Matthias Studer and Reto Bürgin (alternative counting methods) (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
