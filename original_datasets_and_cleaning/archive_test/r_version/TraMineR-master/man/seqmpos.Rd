\name{seqmpos}
\alias{seqmpos}
\title{Number of matching positions between two sequences.}
\description{
Returns the number of common elements, i.e., same states appearing at the same position in the two sequences.
}
\usage{
 seqmpos(seq1, seq2, with.missing=FALSE)
}
\arguments{
  \item{seq1}{a sequence from a sequence object.}
  \item{seq2}{a sequence from a sequence object.}
  \item{with.missing}{if \code{TRUE}, gaps appearing at the same position in both sequences are also considered as common elements}.
}
\seealso{
 \code{\link{seqLLCP}}, \code{\link{seqLLCS} }.
}
\examples{
data(famform)
famform.seq <- seqdef(famform)

seqmpos(famform.seq[1,],famform.seq[2,])
seqmpos(famform.seq[2,],famform.seq[4,])

## Example with gaps in sequences
a <- c(NA,"A",NA,"B","C")
b <- c(NA,"C",NA,"B","C")

ex1.seq <- seqdef(rbind(a,b))

seqmpos(ex1.seq[1,], ex1.seq[2,])
seqmpos(ex1.seq[1,], ex1.seq[2,], with.missing=TRUE)

}

\author{Alexis Gabadinho (with Gilbert Ritschard for help page)}

\keyword{Dissimilarity measures}
