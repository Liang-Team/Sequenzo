\name{seqipos}
\alias{seqipos}
%
\author{Gilbert Ritschard}
%
\title{Indicators for binary sequences of positive and negative states.}
%
\description{
  Indicators for the underlying binary sequences of positive and negative states.
  Possible indicators are the proportion of positive states within each sequence, i.e. of positive spells if computed on the sequences of distinct successive states (DSS), objective volatility of the binary sequences, and capacity to integrate a positive state.
}
\usage{
  seqipos(seqdata, dss=NULL, pos.states=NULL, neg.states=NULL, index="share",
     pow=1, w=.5, with.missing=FALSE)
}
\arguments{
  \item{seqdata}{a state sequence object (\code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{dss}{logical. Should the proportion be computed inside the DSS sequences? Defaults to \code{TRUE} when \code{index="share"} and to \code{FALSE} otherwise.}

  \item{pos.states}{vector of positive states.}

  \item{neg.states}{vector of negative states.}

  \item{index}{character string. One of \code{"share"} (proportion of positive states or spells), \code{"volatility"} (objective volatility), or \code{"integr"} (capability to integrate a positive state). See Details. Default is \code{"share"}.}

  \item{pow}{real. Recency exponent passed to \code{\link{seqintegr}}. Only used when \code{index="integr"}. Default is 1.}

  \item{w}{real in range [0,1]. Relative weight of proportion of visited states in volatility (see \code{\link{seqivolatility}}). Only used when \code{index="volatility"}. Default is .5.}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}
}
\details{
  The function transforms the provided state sequences into binary sequences of positive and negative states. When \code{dss = TRUE}, the counts of positive and negative elements give the number of positive and negative spells. The binary state sequence object is returned as an attribute.

  When \code{neg.states=NULL}, states not listed on the \code{pos.states} argument are considered as negative and conversely when \code{pos.states=NULL}. When \code{with.missing=TRUE}, the missing state \code{nr} will be considered as positive if included in the \code{pos.states} list or as negative if in \code{neg.states}.  When \code{with.missing=FALSE} (default) missing states are ignored. However, when missing values appear within a spell, such as in AA*ABB, the spell is split into two consecutive spells in a same state. For the example we would have AAB, i.e. if A is positive and B negative, a proportion of 2/3 of positive spells.

  When both \code{pos.states} and \code{neg.states} are provided, states of the alphabet (including the \code{nr} code when \code{with.missing=TRUE}) that belong nor to \code{pos.states} nor to \code{neg.states} are ignored.

  For \code{index="share"}, letting \eqn{npos} be the number of positive states and \eqn{nneg} the number of negative states in the sequence, the function returns the value of \eqn{npos/(npos + nneg)} for each sequence. With \code{dss=TRUE}, this is the normative volatility of Brzinsky-Fay (2007,2018).

  For \code{index="volatility"}, the function returns the objective volatility of the binary sequences. See \code{\link{seqivolatility}}.

  For \code{index="integr"}, the function returns the index of integrative potential for the positive state. See \code{\link{seqintegr}}.
  }

\value{
  A numeric vector of \cr
  - when \code{index="share"}, the proportion of positive states (or spells) within each sequence;\cr
  - when \code{index="volatility"}, the objective volatility of the binary sequences;\cr
  - when \code{index="integration"}, the index of integration into a positive state.\cr
  The binary sequence as an attribute \code{sbinary}.
}

\seealso{
\code{\link{seqindic}, \link{seqintegr}, \link{seqivolatility}}
}
\references{
Brzinsky-Fay, C. (2007) Lost in Transition? Labour Market Entry Sequences of School Leavers in Europe, \emph{European Sociological Review}, 23(4). \doi{10.1093/esr/jcm011}

Brzinsky-Fay, C. (2018) Unused Resources: Sequence and Trajectory Indicators. International Symposium on Sequence Analysis and Related Methods, Monte Verita, TI, Switzerland, October 10-12, 2018.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}

\examples{
data(ex1)
sx <- seqdef(ex1[,1:13], right="DEL")
nr <- attr(sx,'nr') ## code for missing values

seqipos(sx, pos.states=c("A","B"))
seqipos(sx, pos.states=c("A","B"), with.missing=TRUE)

## ignoring state "D"
seqipos(sx, pos.states=c("A","B"), neg.states=c("C",nr), with.missing=TRUE)

seqipos(sx, pos.states=c("A","B"), dss=FALSE)
seqipos(sx, pos.states=c("A","B",nr), dss=FALSE, with.missing=TRUE)

seqipos(sx, pos.states=c("A","B"), index="volatility")
seqipos(sx, pos.states=c("A","B"), index="integr")

## retrieving the binary sequences
ip <- seqipos(sx, pos.states=c("A","B"))
attr(ip,"sbinary")
ip <- seqipos(sx, pos.states=c("A","B"), with.missing=TRUE)
attr(ip,"sbinary")

}

\keyword{Longitudinal characteristics}
