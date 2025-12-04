\name{seqsubsn}
\alias{seqsubsn}
\title{Number of distinct subsequences in a sequence.}
\description{
Computes the number of distinct subsequences in a sequence using Elzinga's algorithm.
}
\details{
The function first searches for missing states in the sequences and if found, adds the missing state to the alphabet for the extraction of the distinct subsequences. A missing state in a sequence is considered as the occurrence of an additional symbol of the alphabet, and two or more consecutive missing states are considered as two or more occurrences of the same state. The \code{with.missing=TRUE} argument is used for calling the \code{\link{seqdss}} function when \code{DSS=TRUE}.
}
\usage{
 seqsubsn(seqdata, DSS=TRUE, with.missing=FALSE)
}
\arguments{
  \item{seqdata}{a state sequence object as defined by the \code{\link{seqdef}} function.}
  
  \item{DSS}{if \code{TRUE}, the sequences of Distinct Successive States (DSS, see \code{\link{seqdss}}) are first extracted (e.g., the DSS contained in \code{'D-D-D-D-A-A-A-A-A-A-A-D'} is \code{'D-A-D'}), and the number of distinct subsequences in the DSS is computed. If \code{FALSE}, the number of distinct subsequences is computed from sequences as they appear in the input sequence object. Hence the number of distinct subsequences is in most cases much higher with the \code{DSS=FALSE} option.}
      
	\item{with.missing}{logical: should non-void missing values be treated as a regular state? 
      If \code{FALSE} (default) missing values are ignored.}
      
}
\value{Vector with the number of distinct subsequences for each sequence in the input state sequence object.}
\seealso{
 \code{\link{seqdss}}.
}
\examples{
data(actcal)
actcal.seq <- seqdef(actcal,13:24)

## Number of subsequences with DSS=TRUE
seqsubsn(actcal.seq[1:10,])

## Number of subsequences with DSS=FALSE
seqsubsn(actcal.seq[1:10,],DSS=FALSE)
}

\author{Alexis Gabadinho (with Gilbert Ritschard for the help page)}

\keyword{Longitudinal characteristics}
