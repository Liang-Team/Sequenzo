\name{seqindic}
\alias{seqindic}
%
\author{Gilbert Ritschard}
%
\title{Sequence indicators}
%
\description{
  Table of per sequence values of selected indicators.
}
\usage{
seqindic(seqdata, indic=c("visited","trans","entr","cplx","turb2n"),
    with.missing=FALSE, ipos.args=list(), prec.args=list(), w=.5)
}
\arguments{
  \item{seqdata}{a state sequence object (class \code{stslist}) as returned by \code{\link[TraMineR]{seqdef}}.}

  \item{indic}{vector of character strings. List of selected indicators among \code{"lgth"} (sequence length), \code{"nonm"} (number of non-missing elements), \code{"dlgth"} (number of spells), \code{"visited"} (number of visited states), \code{"visitp"} (proportion of states visites), \code{"trans"} (number of state changes), \code{"transp"} (number of state changes as a proportion of maximum possible transitions), \code{"meand"} (mean spell duration),  \code{"dustd"} (duration standard deviation),
      \code{"meand2"} (mean spell duration taking non-visited states into account), \code{"dustd2"} (duration standard deviation taking non-visited states into account),
      \code{"entr"} (longitudinal normalized entropy), \code{"nsubs"} (number of subsequences of the DSS sequence), \code{"volat"} (objective volatility), \code{"cplx"} (complexity index),
      \code{"turb"} (turbulence), \code{"turbn"} (normalized turbulence),
      \code{"turb2"} (turbulence taking non-visited states into account), \code{"turb2n"} (normalized turbulence taking non-visited states into account),
      \code{"ppos"} (proportion of positive states), \code{"nvolat"} (normative volatility), \code{"vpos"} objective volatility of positive-negative state sequences, \code{"integr"} (integrative potential),  \code{"degrad"} (degradation), \code{"bad"} (badness), \code{"prec"} (precarity index),  \code{"insec"} (insecurity). Can also be \code{"all"}, \code{"basic"}, \code{"diversity"}, \code{"complexity"},  \code{"binary"}, and \code{"ranked"}, each selecting a subset of the indicators. Setting \code{indic="all"} selects all indicators but \code{"ppos"}, \code{"vpos"}, \code{"nvolat"}, \code{"integr"}, \code{"degrad"}, \code{"bad"}, \code{"prec"}, and \code{"insec"} that require an additional argument list.}

	\item{with.missing}{logical: should non-void missing values be treated as a regular state? If \code{FALSE} (default) missing values are ignored.}

  \item{ipos.args}{list: when any of \code{"ppos"}, \code{"nvolat"}, \code{"vpos"}, or \code{"integr"} is requested, the arguments passed to \code{seqipos} together with \code{seqdata} and \code{with.missing}. See \code{\link{seqipos}}}

  %\item{integr.args}{list: when \code{"integr"} requested, the arguments passed to \code{seqintegr} together with \code{seqdata} and \code{with.missing}. See \code{\link{seqintegr}}.}

  \item{prec.args}{list: when any of \code{"degrad"}, \code{"bad"}, \code{"prec"}, or \code{"insec"} is requested, the arguments passed to the underlying functions together with \code{seqdata} and \code{with.missing}. See \code{\link{seqprecarity}} and \code{\link{seqibad}}.}

  \item{w}{real in range [0,1]: when \code{volat} weight coefficient passed to \code{\link{seqivolatility}}. Default is .5.}
}
\details{
  The number of visited states is the number of different elements in the sequence, e.g. 2 for \code{aababba}. The recurrence index \code{'recu'} is the average number of visits of visited states, i.e. \code{Dlgth/Visited}, the number of spells on the number of visited states.

  The sequence length, number of transitions, longitudinal entropy, duration standard deviation, volatility, complexity, turbulence, degradation, badness, precarity, and insecurity are computed respectively with functions \code{\link{seqlength}}, \code{\link{seqtransn}}, \code{\link{seqient}}, \code{\link{seqivardur}},  \code{\link{seqivolatility}}, \code{\link{seqici}}, \code{\link{seqST}}, \code{\link{seqidegrad}}, \code{\link{seqibad}},  \code{\link{seqprecarity}}, and  \code{\link{seqinsecurity}}. The proportion of positive states, normative volatility, and integrative potential are computed with \code{\link{seqipos}}. See corresponding help pages for details.

  The proportion of positive states (\code{'ppos'}) and the normative volatility (\code{'nvolat'}) are the proportions of positive elements in respectively the original sequences and the DSS. They ignore the value of \code{dss} in the \code{ipos.args} list.

  The \code{with.missing} argument applies to all indicators but the length. \code{'lgth'} returns the length obtained with \code{with.missing=TRUE}, and \code{'nonm'} the length obtained with \code{with.missing=FALSE}.
}
\value{
  A data frame with the selected indicators. Names are:
  \cr \code{Lght}: Length of the sequence
  \cr \code{NonM}: Number of non-missing elements
  \cr \code{Dlgth}: Number of spells (length of DSS)
  \cr \code{Visited}: Number of visited states
  \cr \code{Visitp}: Proportion of states visited
  \cr \code{Recu}: Recurrence: average number of visits to visited states
  \cr \code{Trans}: Number of transitions (state changes)
  \cr \code{Transp}: Number of state changes as a proportion of maximum number of transitions
  \cr \code{Entr}: Longitudinal entropy
  \cr \code{Meand}: Mean spell duration
  \cr \code{Dustd}: Duration standard deviation
  \cr \code{Meand2}: Mean spell duration taking non visited states into account
  \cr \code{Dustd2}: Duration standard deviation taking non visited states into account
  \cr \code{Nsubs}: Number of subsequences of the DSS sequence
  \cr \code{Volat}: Objective volatility
  \cr \code{Cplx}: Complexity
  \cr \code{Turb}: Turbulence
  \cr \code{Turbn}: Normalized turbulence
  \cr \code{Turb2}: Turbulence taking non visited states into account
  \cr \code{Turbn2}: Normalized turbulence taking non visited states into account
  \cr \code{Ppos}: Proportion of positive states
  \cr \code{Nvolat}: Normative volatility (proportion of positive spells)
  \cr \code{Vpos}: Objective volatility of positive-negative state sequences
  \cr \code{Integr}: Integrative capacity (potential)
  \cr \code{Degrad}: Degradation
  \cr \code{Bad}: Badness
  \cr \code{Prec}: Precarity
  \cr \code{Insec}: Insecurity
}
\seealso{
\code{\link{seqlength}}, \code{\link{seqtransn}}, \code{\link{seqient}}, \code{\link{seqivardur}},  \code{\link{seqivolatility}}, \code{\link{seqici}}, \code{\link{seqST}}, \code{\link{seqidegrad}}, \code{\link{seqibad}},  \code{\link{seqprecarity}},  \code{\link{seqinsecurity}}, \code{\link{seqipos}}.
}


\references{
Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}


\examples{
data(ex1)
sx <- seqdef(ex1[,1:13], right="DEL")
print(sx, format='SPS')
seqindic(sx, indic=c("lgth","nonm","visited","turbn","cplx"))
seqindic(sx, indic=c("lgth","nonm","visited","turbn","cplx"), with.missing=TRUE)
seqindic(sx, indic=c("lgth","dlgth","ppos","integr","prec"), with.missing=TRUE,
          ipos.args=list(pos.states=c("A","B")),
          prec.args=list(state.order=c("A","B","C"), state.equiv=list(c("C","D"))))
seqindic(sx, indic=c("volat","binary"), ipos.args=list(pos.states=c("A","B")))
seqindic(sx, indic=c("basic","integr"), ipos.args=list(pos.states="D"))
}
\keyword{Longitudinal characteristics}
