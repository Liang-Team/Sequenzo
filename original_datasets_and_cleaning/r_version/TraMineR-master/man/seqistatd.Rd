\name{seqistatd}
\alias{seqistatd}
\title{State frequencies in each individual sequence}
\description{
  Returns the state frequencies (total durations) for each sequence in the sequence object.
}
\usage{
 seqistatd(seqdata, with.missing=FALSE, prop=FALSE)
}
\arguments{
  \item{seqdata}{a sequence object (see \code{\link{seqdef}} function).}
  \item{with.missing}{logical: if set as \code{TRUE}, total durations are also computed for the missing status (gaps in the sequences). See \link{seqdef} on options for handling missing values when creating sequence objects.}
  \item{prop}{logical: if \code{TRUE}, proportions of time spent in each state are returned instead of absolute values. This option is specially useful when sequences contain missing states, since the sum of the state durations may not be the same for all sequences.}
}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}

%\seealso{}

\examples{
data(actcal)
actcal.seq <- seqdef(actcal,13:24)
seqistatd(actcal.seq[1:10,])

## Example using "with.missing" argument
data(ex1)
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)

seqistatd(ex1.seq)
seqistatd(ex1.seq, with.missing=TRUE)

}

\author{Alexis Gabadinho}

\keyword{Longitudinal characteristics}
\keyword{State sequences} 

