\name{seqLLCP}
\alias{seqLLCP}
\title{Compute the length of the longest common prefix of two sequences}
\description{
Returns the length of the longest common prefix of two sequences. This attribute is described in \cite{Elzinga (2008)}.
}
\usage{
seqLLCP(seq1, seq2)
}
\arguments{
  \item{seq1}{a sequence from a sequence object.}
  \item{seq2}{a sequence from a sequence object.}
}

\value{an integer being the length of the longest common prefix of the two sequences.}
\seealso{\code{\link{seqdist}}}

\references{
Elzinga, Cees H. (2008). Sequence analysis: Metric representations of categorical time
series. \emph{Technical Report}, Department of Social Science Research Methods, Vrije Universiteit, Amsterdam.
}

\examples{
data(famform)
famform.seq <- seqdef(famform)

## The LCP's length between sequences 1 and 2
## in the famform sequence object is 2
seqLLCP(famform.seq[1,],famform.seq[2,])
}
\keyword{Dissimilarity measures}
