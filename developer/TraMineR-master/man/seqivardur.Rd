\name{seqivardur}
\alias{seqivardur}
\alias{print.seqivardur}
%
\author{Gilbert Ritschard}
%
\title{Variance of spell durations in individual sequences}
%
\description{
  Variance of spell durations of individual state sequences.
}
\usage{
seqivardur(seqdata, type=1, with.missing=FALSE)

\method{print}{seqivardur}(x, stat='var', ...)
}
\arguments{
  \item{seqdata}{a state sequence object (\code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{type}{either 1 or 2. The default \code{type=1} ignores non visited states. Type 2 takes into account the 0-time spent in non-visited states (see Details).}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}

  \item{x}{an outcome of \code{seqivardur}.}

  \item{stat}{string or vector of strings. Duration statistic to be printed. Either \code{'mean'} (mean duration), \code{'std'} (standard deviation), \code{'var'} (variance), \code{'vmax'} (maximum variance for number of spells), or \code{'all'}. Default is \code{'var'}.}

  \item{...}{further arguments such as \code{digits} passed to the next \code{\link{print}} method.}
}
\details{
  The returned variance is a population variance, i.e. the sum of squares divided by the number of terms.

  When \code{type=1}, this is the variance of the observed spells in the sequence. When \code{type=2}, in addition to the observed spells one spell of length 0 is considered for each non-visited state. The mean duration is computed the same way.
}

\value{
  A numeric vector with the variance of spell duration of each sequence.

  The returned object has two attributes:
   \item{vmax}{the maximum value that the variance of each sequence could reach given its number of spells.}
   \item{meand}{the mean spell duration of each sequence.}
}

\seealso{
\code{\link{seqdur}}, \code{\link{seqindic}}
}
\references{
Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}


\examples{
sq.dat <- c('AAAA','AAAC','ABC','ABAA','AC')
sq <- seqdef(seqdecomp(sq.dat, sep=''), right=NA)

seqivardur(sq, type=1)
seqivardur(sq, type=1, with.missing=TRUE)
vd2 <- seqivardur(sq, type=2)
print(vd2, stat='all')
vd2m <- seqivardur(sq, type=2, with.missing=TRUE)
print(vd2m, stat=c('var','vmax'))

}

\keyword{Longitudinal characteristics}
