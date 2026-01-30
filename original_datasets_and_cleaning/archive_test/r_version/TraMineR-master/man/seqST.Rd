\name{seqST}
\alias{seqST}
\title{Sequences turbulence}
\description{
  Elzinga's turbulence for each sequence in a sequence data set.
}

\usage{
 seqST(seqdata, norm=FALSE, silent=TRUE, with.missing=FALSE, type=1)
}
\arguments{
  \item{seqdata}{a state sequence object as returned by the the \code{\link{seqdef}} function.}

  \item{norm}{logical: should the turbulence index be normalized?}

  \item{silent}{logical: should messages about running operations (extracting dss and durations, computing turbulence) be displayed?}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}

  \item{type}{either 1 or 2. Type of duration variance. The default \code{type=1} ignores non visited states. Type 2 takes into account the 0-time spent in non-visited states  (see \code{\link{seqivardur}}).
      }

}


\details{
 Sequence turbulence is a measure proposed by \cite{Elzinga
& Liefbroer (2007)}. It is based on the number \eqn{\phi(x)}{phi(x)}
of distinct subsequences that can be extracted from the distinct
successive state (DSS) sequence and the variance of the consecutive times \eqn{t_i}
spent in the distinct states. For a sequence \eqn{x}, the formula is

  \deqn{T(x)=\log_{2}(\phi(x)\,\frac{s_{t,max}^2(x) + 1}{s_t^2(x) + 1})}{%
        T(x) = log_2 [phi(x) (s_{t,max}^2(x) + 1)/(s_t^2(x) + 1)]}

where \eqn{s_t^2(x)} is the variance of the successive state
durations in sequence \eqn{x} and \eqn{s_{t,max}^2(x)} is the maximum
value that this variance can take given the number of spells and the total
duration of the sequence. For \code{type=1}, this maximum is computed as

\deqn{s_{t,max}^2 =(d-1)(1-\bar{t})^2}{%
      s_{t,max}^2 = (d-1)(1- bar{t})^2}

 where \eqn{\bar{t}}{bar{t}} is the mean consecutive time spent in the
distinct states, i.e. the sequence duration \eqn{t} divided by the number
\eqn{d} of distinct states in the sequence.
For \code{type=2}, the variance takes into account the 0-time spent in non-visited states and the maximum is adjusted for the maximum number of non-visited states for the number of spells (see \cite{Ritschard, 2021}).

When \code{with.missing=TRUE}, the function searches for missing states in the sequences and if found, adds the missing state to the alphabet for the computation of the turbulence. In this case the \code{\link{seqdss}} and \code{\link{seqdur}} functions for extracting the distinct successive state sequences and the associated durations are called with the \code{{with.missing=TRUE}} argument. Thus, a missing state in a sequence is considered as the occurrence of an additional symbol of the alphabet and two or more consecutive missing states are considered as two or more occurrences of this additional state. E.g. the DSS of \code{A-A-*-*-*-B-B-C-C-D} is \code{A-*-B-C-D} and the associated durations are \code{2-3-2-2-1}.

The normalized value is obtained by subtracting 1 to the index and then dividing by the resulting value for a sequence made by the successive repetition of the alphabet up to the maximal length in \code{seqdata} (\cite{Ritschard, 2021)}).
}

\value{a sinlge-column matrix of length equal to the number of sequences in
\code{seqdata} containing the turbulence value of each sequence. Normalized values are returned when \code{norm=TRUE}.}

\seealso{
 \code{\link{seqdss}}, \code{\link{seqdur}}, \code{\link{seqsubsn}}. For alternative measures of sequence complexity see \code{\link{seqivolatility}},  \code{\link{seqici}}, \code{\link{seqindic}}.
}
\references{
  Elzinga, Cees H. and Liefbroer, Aart C. (2007).
  De-standardization of Family-Life Trajectories of Young Adults: A Cross-National Comparison Using Sequence Analysis.
  \emph{European Journal of Population}, 23, 225-250.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}
\examples{
  ## Loading the 'actcal' example data set
  data(actcal)
  ## Here we consider only the first 10 sequences
  actcal <- actcal[1:10,]

  ## Defining a sequence object with data in columns 13 to 24
  ## (activity status from January to December 2000)
  actcal.seq <- seqdef(actcal[,13:24], informat='STS')

  ## Computing the sequences turbulence
  turb <- seqST(actcal.seq)

  ## Normalized turbulence
  turb.norm <- seqST(actcal.seq, norm=TRUE)

  ## Normalized turbulence taking non-visited states into account.
  turb2.norm <- seqST(actcal.seq, norm=TRUE, type=2)

}

\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Longitudinal characteristics}
