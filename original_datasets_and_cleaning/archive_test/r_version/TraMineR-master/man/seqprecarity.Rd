\encoding{utf8}

\name{seqprecarity}

\alias{seqidegrad}
\alias{seqprecarity}
\alias{seqinsecurity}
%%\alias{seqprecarity.old}
\alias{seqprecorr}
\alias{print.seqidegrad}
\alias{print.seqprec}
%\alias{print.seqprecorr}

\title{Degradation, Precarity, and Insecurity indexes}

\description{
The degradation index returned by \code{seqidegrad} is a normalized transformation of the difference between the proportion of downward and upward state changes (transitions).
The precarity and insecurity indexes are composite measures of complexity, degradation tendency, and undesirableness of the first state in the sequence.
}

\usage{
seqidegrad(seqdata, state.order=alphabet(seqdata, with.missing), state.equiv=NULL,
    stprec=NULL, with.missing=FALSE,
    penalized="BOTH", method="RANK", weight.type="ADD",
    pow=1, border.effect=10)

seqprecarity(seqdata, correction=NULL,
    state.order=alphabet(seqdata, with.missing), state.equiv=NULL,
    stprec=NULL, with.missing=FALSE,
    otto=.2, a=1, b=1.2, method = "TRATEDSS",
    ...)

seqinsecurity(seqdata, correction=NULL,
    state.order=alphabet(seqdata, with.missing), state.equiv=NULL,
    stprec=NULL, with.missing=FALSE,
    pow = 1, spow=pow, bound=FALSE, method = "RANK",
    ...)

seqprecorr(...)
}

\arguments{
  \item{seqdata}{a state sequence object (class \code{stslist}) as returned by the \code{\link[TraMineR]{seqdef}} function.}

  %\item{type}{Integer: 1 or 2. \code{type = 1} is the precarity index of \cite{Ritschard et al (2018)} and \code{type = 2} is the insecurity index defined in \cite{Ritschard (2021)}.}

  \item{correction}{Vector of non-negative correction factor values. If \code{NULL} (default), the correction factor is set as the degradation value returned by \code{seqidegrad}. See details.}

  \item{state.order}{Vector of short state labels defining the order of the states. First the less precarious (most positive) state and then the other states in increasing precariousness order. States of the alphabet that are not included here (and are not equivalent to one of the listed state) define the non-comparable states.}

  \item{state.equiv}{List of state equivalence classes. Each class in the list is given as the vector of the short labels of the states forming the class. }

  \item{stprec}{Vector of state undesirableness degrees. If \code{NULL} the values are derived from the state order using \code{\link{seqprecstart}}. If not \code{NULL}, the values in \code{stprec} should conform the order of the alphabet.
      Use negative values to indicate non-comparable states. See details.}

  \item{with.missing}{Logical. Should the missing state be considered as an element of the alphabet?}

  \item{otto}{Scalar in the range [0,1]. Trade-off weight between the precarity degree of the initial state and the corrected complexity. Default is \code{otto=.2}}

  \item{a}{Non-negative real value. Exponent weight of the complexity. Default is 1.}

  \item{b}{Non-negative real value. Exponent weight of the correction factor. Default is 1.2. }

  \item{penalized}{One of \code{'BOTH'} (default), \code{'NEG'}, \code{'POS'}, or \code{'NO'}. What should be penalized or rewarded? \code{'NEG'} only negative transitions, \code{POS} only positive transitions (with negative penalization), \code{'BOTH'} penalize negative transitions and reward positive ones, \code{NO} no penalization. Can also be logical with \code{TRUE} equivalent to \code{'BOTH'} and \code{FALSE} equivalent to \code{'NO'}.}

  \item{method}{One of \code{'FREQ'}, \code{'FREQ+'}, \code{'TRATE'}, \code{'TRATE+'}, \code{'TRATEDSS'} (default for \code{seqprecarity} for backward compatibility), \code{'TRATEDSS+'}, \code{'RANK'} (default for \code{seqinsecurity} and \code{seqidegrad}), \code{'RANK+'}, or \code{'ONE'}. Method for determining transition weights. Weights based on transition probabilities: \code{'FREQ'} overall frequency of the transitions, \code{'TRATE'} transition probabilities, and \code{'TRATEDSS'} transition probabilites in the DSS sequences. \code{'RANK'} differences between state undesirableness degrees. \code{'ONE'} no weight. With the \code{+} form the returned penalty is adjusted by the mean transition weight in the sequence.}

  \item{weight.type}{One of \code{'ADD'} (default), \code{'INV'}, or \code{'LOGINV'}. When \code{method} is one of \code{'FREQ'}, \code{'TRATE'} or \code{'TRATEDSS'}, how weights are derived from the transition probabilities: \code{'ADD'} additive (1-p), \code{'INV'} inverse (1/p), and \code{'LOGINV'} log of inverse. Ignored when any other \code{method} is selected.}

  \item{pow}{Real or logical. Recency weight exponent for potential to integrate the next spell. If real, transition weights are  adjusted by the potential to integrate the next state using the \code{pow} value. \code{pow=TRUE} is equivalent to \code{pow=1}. (See details.)}

  \item{spow}{Real. Recency weight exponent for potential to integrate the first spell.}

  \item{bound}{Logical. Should the insecurity index be bounded by undesirableness degrees of best and worst states in the sequence?}

  \item{border.effect}{Real. Value (strictly greater than 1) used to adjust estimated transition probabilities to avoid border effect. Default is 10. See details.}

  \item{...}{Arguments passed to \code{seqidegrad} when \code{correction=NULL}}
}


\details{
The \code{seqidegrad} function returns for each sequence \eqn{x} the difference \eqn{q(x)} between the proportions of downward and upward transitions (state changes).

The argument \code{penalized} allows to chose between three strategies for computing \eqn{q(x)}: only penalizing negative weights (in which case \eqn{q(x)} is the proportion of negative transitions), only rewarding (with negative penalties) positive transitions, and applying both positive and negative penalties. The transitions can be weighted and the type of transition weights used is selected with the \code{method} argument. For weights based on transition probabilities, the way how theses probabilites are transformed into weights is controlled with \code{weight.type}. To avoid a border effect, when any computed transition probability \eqn{p} is close from 1 (\eqn{p > 1 - .1/d}), all \eqn{p}'s are adjusted as \eqn{p - p/d}, where \eqn{d} is the \code{border.effect} parameter. With \code{method="RANK"}, the weights are set as the differences between the to and from state undesirableness.
When \code{pow} is not \code{FALSE}, the weight of each transition is multiplied by the potential to integrate the next spell using the provided \code{pow} value.

The precarity and insecurity indexes of a sequence \eqn{x} are both based on the complexity index (\cite{Gabadinho et al., 2010}) \eqn{c(x)} (See the \code{\link[TraMineR]{seqici}} function) and the undesirableness degree \eqn{\pi(x_1)}{pi(x_1)} of the starting state.

The precarity applies a multiplicative correction to the complexity. It is defined as

 \deqn{prec(x) = \lambda \pi(x_1) + (1-\lambda)(1 + r(x))^\beta c(x)^\alpha}{prec(x) = lambda * pi(x_1) + (1 - lambda)*(1 +r(x))^b*c(x)^a}

where \eqn{r(x)} is the correction factor (argument \code{correction}) for the sequence. The \eqn{\lambda}{lambda} parameter (argument \code{otto}) determines the trade-off between the importance of the undesirableness of the starting state and of the corrected complexity index. Parameters \eqn{\alpha}{a} and \eqn{\beta}{b} (arguments \code{a} and \code{b}) are exponent weights of respectively the complexity and the correction factor.

The insecurity index applies an additive correction of the complexity:

  \deqn{insec(x) = \pi(x_1) integr(x,sp1) + r(x) + c(x)}{insec(x) = pi(x_1) integr(x,sp1) + r(x) + c(x)}

where \eqn{integr(x,sp1)} is the potential to integrate the first spell (proportion of sequence length covered by first spell when \code{spow=0}).

When \code{correction = NULL} (default), \eqn{r(x)} is set as the degradation index \eqn{q(x)} provided by \code{seqidegrad}. The degradation is computed with \code{pow=FALSE} for the precarity and using the provided \code{pow} value for the insecurity.

When \code{stprec} is a vector, negative values indicate non-comparable sates that receive each the mean positive undesirableness value. After this transformation, the vector is normalized such that the minimum is 0 and the maximum 1.

When equivalent classes are provided, the class mean undesirableness degree is assigned to each state of the class (see \code{\link{seqprecstart}}). For the count of transitions a same state value is assigned to all equivalent states.

Non-comparable states (those not listed on the \code{state.order} argument and not equivalent to a listed state) all receive the mean undesirableness value. For the count of transitions, transitions from and to non-comparable states are ignored and replaced by a transition between the states that immediately precede and follow a spell in non-comparable states.

When there are missing states in the sequences, set \code{with.missing = TRUE} to treat the missing state as an additional state. In that case the missing state will be considered as non-comparable unless you include the \code{nr} attribute of \code{seqdata} in \code{state.order} or \code{state.equiv}. With \code{with.missing = FALSE}, transitions to and from the missing state will just be ignored and the undesirableness value of the first valid state will be used as starting undesirableness.
%To drop these starting missing states, use \code{left='DEL'} in \code{\link{seqdef}} when defining the state sequence object.

The earlier \code{seqprecorr} function is obsolete, use \code{seqidegrad} with \code{pow=FALSE} and \code{method='TRATEDSS'} instead.
}



\value{

For \code{seqprecarity} and \code{seqinsecurity}, an object of class \code{seqprec} with the value of the precarity or insecurity index for each sequence. The returned object has an attribute \code{stprec} that contains the state precarity degree used at the starting position. The associated print method (\code{print.seqprec}) prints the state precarity values without the additional attribute.


For \code{seqidegrad} %and \code{seqprecorr},
an object of class \code{seqidegrad} with the degradation index \eqn{q(x)}
%for \code{seqprecorr} or the degradation index \eqn{(1-q(x))/2} for \code{seqidegrad}
and as additional attributes: \code{tr} the used transition weights; \code{signs} the transitions signs; \code{state.noncomp} the non-comparable states; and \code{state.order} the used state order. The associated print method (\code{print.seqidegrad}) prints the outcome values without the additional attributes.

}

\seealso{
 \code{\link{seqici}}, \code{\link{seqibad}}, \code{\link{seqprecstart}}, \code{\link{seqindic}}.
}

\references{
Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.

Ritschard, G., Bussi, M., and O'Reilly, J. (2018), "An index of precarity for
measuring early employment insecurity", in G. Ritschard, and M. Studer, \emph{Sequence Analysis and Related Approaches: Innovative Methods and Applications}, Series Life Course Research and Social
Policies, Vol. 10, pp 279-295. Cham: Springer, \doi{10.1007/978-3-319-95420-2_16}.

Gabadinho, A., Ritschard, G., Studer, M. and Müller, N.S. (2010), "Indice de complexité pour le tri et la comparaison de séquences catégorielles", In Extraction et gestion des connaissances (EGC 2010), \emph{Revue des nouvelles technologies de l'information RNTI}. Vol. E-19, pp. 61-66.

}

\examples{
## Defining a sequence object with columns 13 to 24
## in the 'actcal' example data set
data(actcal)
actcal <- actcal[1:20,] ## Here, only a subset
actcal.seq <- seqdef(actcal[,13:24], alphabet=c('A','B','C','D'))

## insecurity and precarity using the original state order
insec <- seqinsecurity(actcal.seq)
prec <- seqprecarity(actcal.seq)
ici <- seqici(actcal.seq) ## complexity

seqn <- seqformat(actcal.seq, to="SPS", compress=TRUE)
tab <- data.frame(seqn,ici,insec,prec)
names(tab) <- c("seq","ici","insec","prec")
tab

## Assuming A and B as equivalent regarding insecurity
insec2 <- seqinsecurity(actcal.seq, state.equiv=list(c('A','B')))
tab <- cbind(tab,insec2)
names(tab)[ncol(tab)] <- "insec2"

## and letting C be non-comparable
insec3 <- seqinsecurity(actcal.seq, state.order=c("A","B","D"),
  state.equiv=list(c('A','B')))
tab <- cbind(tab,insec3)
names(tab)[ncol(tab)] <- "insec3"

## bounding insecurity with undesirableness of best and worst state in sequence
insec4 <- seqinsecurity(actcal.seq, state.order=c("A","B","D"),
  state.equiv=list(c('A','B')), bound=TRUE)
tab <- cbind(tab,insec4)
names(tab)[ncol(tab)] <- "insec4"

## degrading index
degr <- seqidegrad(actcal.seq, state.order=c("A","B","D"),
                  state.equiv=list(c('A','B')))
tab <- cbind(tab,degr)
names(tab)[ncol(tab)] <- "degr"
tab

## Precarity with transition weights based on differences in state undesirableness
prec.rank <- seqprecarity(actcal.seq, method='RANK')

###############
## Indexes in presence of missing values:
##  missing state treated as an additional state
data(ex1)
## by default right missings are dropped from the sequences
s.ex1 <- seqdef(ex1[,1:13])
state.order=c("A","B","C","D") ## missing left as non-comparable
seqprecarity(s.ex1, state.order=state.order, with.missing=TRUE)
seqinsecurity(s.ex1, state.order=state.order, with.missing=TRUE)

## same result using the correction argument
dgp <- seqidegrad(s.ex1, with.missing=TRUE, state.order=state.order, method='TRATEDSS')
seqprecarity(s.ex1, state.order=state.order, with.missing=TRUE, correction=dgp)

dg <- seqidegrad(s.ex1, with.missing=TRUE, state.order=state.order)
seqinsecurity(s.ex1, state.order=state.order, with.missing=TRUE, correction=dg)

## bounding insecurity with undesirableness of best and worst state in sequence
seqinsecurity(s.ex1, state.order=state.order, with.missing=TRUE, bound=TRUE)


}

\author{Gilbert Ritschard}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Longitudinal characteristics}
