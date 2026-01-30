\name{seqibad}
\alias{seqibad}
%
\author{Gilbert Ritschard}
%
\title{Badness index}
%
\description{
  Badness index of each sequence, i.e. the sum of undesirableness of each state weighted by the potential to integrate that state in the sequence.
}
\usage{
  seqibad(seqdata, pow=1, with.missing=FALSE, ...)
}
\arguments{
  \item{seqdata}{a state sequence object (\code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{pow}{real. Exponent applied to the position in the sequence. Higher value increase the importance of recency (see \code{\link{seqintegration}}). Default is 1.}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}

  \item{...}{arguments such as \code{stprec} or \code{state.order} required by \code{\link{seqprecstart}} to determine/normalize the state undesirableness degrees.}

}
\details{
  For each sequence, the badness is the sum of the undesirableness of each state weighted by the potential to integrate the state. As long as \code{pow} is strictly greater than zero, the undesirableness of states occurring at the end of the sequence get higher weights than those at the beginning. The index reaches its maximum 1 for a sequence made of a single spell in the worst state and the minimum 0 for a sequence made of a single spell is the most favorable state.

}

\value{
  A vector with the badness index for each sequence.
}

\references{
Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}

\seealso{
\code{\link{seqintegr}}, \code{\link{seqidegrad}}, \code{\link{seqprecarity}}
}
\examples{
data(ex1)
sx <- seqdef(ex1[,1:13], right="DEL")

seqibad(sx) ## using original alphabet order
seqibad(sx, stprec=c(1,2,3,6)) ## user defined undesirableness values
seqibad(sx, with.missing=TRUE, state.order=c('A','B','C','D'))

}
\keyword{Longitudinal characteristics}
