\name{seqient}
\alias{seqient}
\title{Within sequence entropies}

\description{Computes normalized or non-normalized within sequence entropies}

\usage{
 seqient(seqdata, norm=TRUE, base=exp(1), with.missing=FALSE, silent=TRUE)
}
\arguments{
  \item{seqdata}{a sequence object as returned by the the \code{\link{seqdef}} function.}

  \item{norm}{logical: should the entropy be normalized? \code{TRUE} by default. (see details)}
  \item{base}{real positive value: base of the logarithm used in the entropy formula (see details). Default is \code{exp(1)}, i.e., the natural logarithm is used.}

  \item{with.missing}{logical: if \code{TRUE}, the missing state (gap in sequences) is handled as an additional state when computing the state distribution in the sequence.}
  \item{silent}{logical: should messages about running operations be displayed?}
}


\details{
The seqient function returns the Shannon entropy of each sequence in \code{seqdata}. The entropy of a sequence is computed using the formula
%
\deqn{h(\pi_1,\ldots,\pi_s)=-\sum_{i=1}^{s}\pi_i\log \pi_i}{h(p_1,...,p_s) = - sum_{i=1}^{s} p_i log(p_i)}
%
where \eqn{s} is the size of the alphabet and \eqn{\pi_i}{p_i} the proportion of occurrences of the \eqn{i}th state in the considered sequence. The base of the log is controlled with the \code{base} argument. Ba default the natural logarithm, i.e. the logarithm in base \eqn{e}, is used. The entropy can be interpreted as the `uncertainty' of predicting the states in a given sequence. If all states in the sequence are the same, the entropy is equal to 0. For example, the maximum entropy for a sequence of length 12 with an alphabet of 4 states is 1.386294 and is attained when each of the four states appears 3 times.

Normalization can be requested with the \code{norm=TRUE} option, in which case the returned value is the entropy divided by the entropy of the alphabet. The latter is an upper bound for the entropy of sequences made from this alphabet. It is exactly the maximal entropy when the sequence length is a multiple of the alphabet size. The value of the normalized entropy is independent of the chosen logarithm base.
}

\value{
a single-column matrix with an entropy value for each sequence in \code{seqdata}; the column length is equal to the number of sequences.
}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009). Mining Sequence Data in \code{R} with the \code{TraMineR} package: A user's guide. \emph{Department of Econometrics and Laboratory of Demography, University of Geneva}.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.

}

\seealso{\code{\link{seqindic}}, \code{\link{seqici}}, \code{\link{seqST}}, and \code{\link{seqstatd}} for the entropy of the cross-sectional state distributions by positions in the sequence.
}

\examples{
data(actcal)
actcal.seq <- seqdef(actcal,13:24)

## Summarize and plots an histogram
## of the within sequence entropy
actcal.ient <- seqient(actcal.seq)
summary(actcal.ient)
hist(actcal.ient)

## Examples using with.missing argument
data(ex1)
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)

seqient(ex1.seq)
seqient(ex1.seq, with.missing=TRUE)
}

\author{Alexis Gabadinho}

\keyword{Longitudinal characteristics}
