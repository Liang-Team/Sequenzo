\name{seqlength}
\alias{seqlength}
\title{Sequence length}
\description{
  Returns the length of sequences.
}
\details{
The length of a sequence is computed by counting its number of non void elements, i.e. including non-void missing values. The \code{seqlength} function returns a vector containing the length of each sequence in the provided sequence object.
}
\usage{
 seqlength(seqdata, with.missing=TRUE)
}
\arguments{
  \item{seqdata}{a sequence object created with the \code{\link{seqdef}} function.}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state?
      Default is \code{TRUE}.
      If \code{FALSE} missing values are considered as void.}
}
\seealso{\code{\link{seqlength.align}}
}
\examples{
## Loading the 'famform' example data set
data(famform)

## Defining a sequence object with the 'famform' data set
ff.seq <- seqdef(famform)

## Retrieving the length of the sequences
## in the ff.seq sequence object
seqlength(ff.seq)
}

\author{Alexis Gabadinho and Gilbert Ritschard}
\keyword{Longitudinal characteristics}
