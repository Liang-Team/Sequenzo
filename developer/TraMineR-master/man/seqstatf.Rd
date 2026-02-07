\name{seqstatf}
\alias{seqstatf}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{State frequencies in the whole sequence data set}
\description{
Overall frequency of each state of the alphabet in the state sequence object.
}
\usage{
seqstatf(seqdata, weighted = TRUE, with.missing=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{a sequence object as defined by the \code{\link{seqdef}} function.}
  \item{weighted}{Logical. Should frequencies account for weights when present in the state sequence object
  (see \code{\link{seqdef}}). Default is \code{TRUE}.}
  \item{with.missing}{Logical. Should non void missing states be treated as regular values? Default is \code{FALSE}.}
}
\details{
The \code{seqstatf} function computes the (weighted) count and frequency of each state of the alphabet in \code{seqdata}, i.e., the (weighted) sum of the occurrences of a state in \code{seqdata}.
}
\value{
A data frame with as many rows as states in the alphabet and two columns, one for the count (Freq) and one for the percentage frequencies (Percent).
}

\seealso{
\code{\link{seqstatd}} for the state distribution by time point (position), \code{\link{seqistatd}} for the state distribution within each sequence.
}

\examples{
## Creating a sequence object from the actcal data set
data(actcal)
actcal.lab <- c("> 37 hours", "19-36 hours", "1-18 hours", "no work")
actcal.seq <- seqdef(actcal, 13:24, labels=actcal.lab)

## States frequencies
seqstatf(actcal.seq)

## Example with weights
data(ex1)
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)

## Unweighted
seqstatf(ex1.seq, weighted=FALSE)

## Weighted
seqstatf(ex1.seq, weighted=TRUE)
}

\author{Alexis Gabadinho}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{State sequences}
\keyword{Global characteristics}

