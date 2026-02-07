\name{seqlength.align}
\alias{seqlength.align}
\title{Align sequence length across domains}
\description{
  Sets lengths of sequences of multiple domains as the shortest lengths across domains.
}
\details{
Sequences in the sequence objects are assumed to be ordered conformably. The length of the \eqn{i}-th sequence in each domain is set as the length of the shortest \eqn{i}-th sequence of the domains. The reduction of length is done by filling end positions with voids.
}
\usage{
 seqlength.align(seq.list)
}
\arguments{
  \item{seq.list}{list of sequence objects (of class \code{stslist}) created with the \code{\link{seqdef}} function. The sequence objects must all have the same number of sequences.}
}
\seealso{\code{\link{seqlength}}}
\examples{
## Using the ex1 data set with sequences of different length
data(ex1)
s1 <- seqdef(ex1[,1:13])
seqlength(s1)

## sequence object s2 with a shorter 1st sequence

s2 <- s1
s2[1,8:13] <- attr(s2,"void")
seqlength(s2)

## aligning sequence lengths
seqlength.align(list(s1,s2))

}

\author{Gilbert Ritschard}
\keyword{Longitudinal characteristics}
