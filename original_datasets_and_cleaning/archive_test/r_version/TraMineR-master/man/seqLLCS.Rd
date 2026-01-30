\name{seqLLCS}
\alias{seqLLCS}
\title{Compute the length of the longest common subsequence of two sequences}
\description{
Returns the length of the longest common subsequence of two sequences. This attribute is described in \cite{Elzinga (2008)}.
}
\usage{
seqLLCS(seq1, seq2)
}
\arguments{
  \item{seq1}{a sequence from a sequence object}
  \item{seq2}{a sequence from a sequence object}
}

\value{an integer being the length of the longest common subsequence of the two sequences.}

\references{
Elzinga, Cees H. (2008). Sequence analysis: Metric representations of categorical time
series. \emph{Technical Report}, Department of Social Science Research Methods, Vrije Universiteit, Amsterdam.
}

\seealso{\code{\link{seqdist}}}

\examples{
LCS.ex <- c("S-U-S-M-S-U", "U-S-SC-MC", "S-U-M-S-SC-UC-MC")
LCS.ex <- seqdef(LCS.ex)
seqLLCS(LCS.ex[1,],LCS.ex[3,])
}
\keyword{Dissimilarity measures}