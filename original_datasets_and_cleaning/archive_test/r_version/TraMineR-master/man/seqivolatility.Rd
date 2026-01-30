\name{seqivolatility}
\alias{seqivolatility}
%
\author{Gilbert Ritschard}
%
\title{Volatility of individual state sequences}
%
\description{
  Returns Brzinsky-Fay's objective volatility of each sequence.
}
\usage{
  seqivolatility(seqdata, w=.5, with.missing=FALSE, adjust=TRUE)
}
\arguments{
  \item{seqdata}{a state sequence object (\code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{adjust}{Logical. Should the indicator be adjusted such that it can reach its bounds 0 and 1. Deafult is \code{TRUE} (see Details).}

  \item{w}{Real in the range [0, 1]. Default is 0.5. Weight given to the proportion of states visited (see Details).}

	\item{with.missing}{Logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}
}
\details{
  The (objective) volatility is the weighted average between the proportion \eqn{pvisited} of states visited and the frequency \eqn{ftrans} of transitions (state changes). Formally,

  \deqn{volatility = w \cdot pvisited + (1-w) \cdot ftrans}{volatility = w * pvisited + (1-w) * ftrans}

  The proportion of states visited is computed as \eqn{(visited - 1)/(|a| - 1}) when \code{adjsut=TRUE} and as \eqn{visited / |a|} when \code{adjsut=FALSE}. Here, \eqn{visited} is the number of states visited and \eqn{|a|} the size of the alphabet.

  The frequency of transition is \eqn{ftrans = \frac{transn}{max.transn}} where
  \eqn{transn} is the number of transitions (state changes) within the sequence, and \eqn{max.transn} the maximum possible transitions in the sequence.

  For the normative volatility, see \code{\link{seqipos}}. For alternative measures of sequence complexity see \code{\link{seqST}},  \code{\link{seqici}}, \code{\link{seqindic}}.
}

\value{
  A numeric vector with the volatility of each sequence.
}

\seealso{
\code{\link{seqintegr}}, \code{\link{seqipos}}, \code{\link{seqindic}}
}
\references{
Brzinsky-Fay, C. Unused Resources: Sequence and Trajectory Indicators. International Symposium on Sequence Analysis and Related Methods, Monte Verita, TI, Switzerland, Oct 10-11, 2018

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.

}


\examples{
data(ex1)
sx <- seqdef(ex1[,1:13], right="DEL")

seqivolatility(sx)
seqivolatility(sx, adjust=FALSE)
seqivolatility(sx, with.missing=TRUE)
seqivolatility(sx, w=.7, with.missing=TRUE)

}

\keyword{Longitudinal characteristics}
