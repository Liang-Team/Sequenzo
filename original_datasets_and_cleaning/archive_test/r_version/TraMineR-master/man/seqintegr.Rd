\name{seqintegr}
\alias{seqintegr}
\alias{seqintegration}
%
\author{Gilbert Ritschard}
%
\title{Integrative potential}
%
\description{
  Returns the index of integrative potential (capability) for each sequence, either a table with the index for each state or a vector with the index for the selected state.
}
\usage{
  seqintegr(seqdata, state=NULL, pow=1, with.missing=FALSE)
}
\arguments{
  \item{seqdata}{a state sequence object (\code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{state}{character string. The state for which to compute the integrative index (see Details). When \code{NULL} the index is computed for each state.}

  \item{pow}{real. Exponent applied to the position in the sequence. Higher value increase the importance of recency (see Details). Default is 1.}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}
}
\details{
  The index of integrative potential or capability \cite{(Brzinsky-Fay, 2007, 2018)} measures the capacity to integrate the selected state within the sequence, i.e. the tendency to reach the selected state and end up in it. The index is defined as the sum of the position numbers occupied by the selected state in the sequence over the sum of all position numbers. Formally, for a sequence \eqn{s} of length \eqn{L}, and numbering the positions \eqn{i} from 1 to \eqn{L}, the index is

  \deqn{integr = \sum_{(i | s_i = state)} i^{pow} / \sum_i i^{pow}}{integr = sum (s_i == state)*i^pow / sum i^pow}

  where \eqn{state} is the selected state. This same index has also been independently developed by \cite{Manzoni and Mooi-Reci (2018)} under the name of quality index.

  The recency exponent \eqn{pow} permits to control the focus given on the latest positions in the sequence. The higher \code{pow}, the higher the importance of the last positions relative to the first ones.

  When \code{with.missing = FALSE}, the index is obtained by using the sum of the positions numbers of the non-missing elements as denominator. To compute the index for the missing state, \code{with.missing} should be set as \code{TRUE}.

  For capability to integrate a set of states see \code{\link{seqipos}}.
}

\value{
  when \code{state=NULL}, a numeric matrix with a row for each sequence and a column by state. When a state is provides, a single column.
}

\references{
Brzinsky-Fay, C. (2007) Lost in Transition? Labour Market Entry Sequences of School Leavers in Europe, \emph{European Sociological Review}, 23(4). \doi{10.1093/esr/jcm011}

Brzinsky-Fay, C. (2018) Unused Resources: Sequence and Trajectory Indicators. International Symposium on Sequence Analysis and Related Methods, Monte Verita, TI, Switzerland, October 10-12, 2018.

Manzoni, A and I. Mooi-Reci (2018) Measuring Sequence Quality, in Ritschard and Studer (eds), \emph{Sequence Analysis and Related Approaches. Innovative Methods and Applications}, Springer, 2018, pp 261-278.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}

\seealso{
\code{\link{seqipos}}, \code{\link{seqivolatility}}, \code{\link{seqindic}}
}
\examples{
data(ex1)
sx <- seqdef(ex1[,1:13], right="DEL")

seqintegr(sx)
seqintegr(sx, with.missing=TRUE)
seqintegr(sx, state="B")
seqintegr(sx, state="B", pow=1.5)

}
\keyword{Longitudinal characteristics}
