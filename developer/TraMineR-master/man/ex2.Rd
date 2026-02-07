\name{ex2}
\docType{data}
\alias{ex2}
\alias{ex2.weighted}
\alias{ex2.unweighted}

\title{Example data sets with weighted and unweighted sequence data}

\description{
Example data sets used to demonstrate the handling of weights. The \code{'ex2.weighted'} data set contains 6 sequences with weights inflating to 100 sequences (sum of weights is 100). The second data frame \code{'ex2.unweighted'} contains the corresponding 100 sequences.

The sequences are, in both data frames, in the \code{'seq'} column, and weights in the \code{'weight'} column of \code{'ex2.weighted'}.

The alphabet is made of four possible states: \code{A, B, C} and \code{D}.
\cr

These data sets are mainly intended to test and illustrate the handling of weights in TraMineR's functions. Weighted results obtained with '\code{ex2.weighted}' data set should be exactly the same as unweighted results obtained with the '\code{ex2.unweighted}' data set.
}

\usage{data(ex2)}
\format{
The command \code{data(ex2)} generates two data frames:
\cr
\code{ex2.weighted}: a data frame with 6 rows, 1 variable containing sequences as character strings, 1 weight variable.
\cr
\code{ex2.unweighted}: a data frame with 100 rows, 1 variable containing sequences as character strings.

}

\examples{
data(ex2)

ex2w.seq <- seqdef(ex2.weighted, 1, weights=ex2.weighted$weight)
ex2u.seq <- seqdef(ex2.unweighted)
}


\source{The brain of the TraMineR package team.}
\author{Gilbert Ritschard and Alexis Gabadinho}

\keyword{Datasets}
