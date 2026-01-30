\name{seqcost}

\alias{seqcost}
\alias{seqsubm}

\title{Generate substitution and indel costs}

\description{
The function \code{seqcost} proposes different ways to generate substitution costs
(supposed to represent state dissimilarities) and possibly indel costs. Proposed methods are:
\code{"CONSTANT"} (same cost for all substitutions), \code{"TRATE"} (derived from the observed transition rates), \code{"FUTURE"} (Chi-squared distance between conditional state distributions \code{lag} positions ahead), \code{"FEATURES"} (Gower distance between state features), \code{"INDELS"}, \code{"INDELSLOG"} (based on estimated indel costs).
The substitution-cost matrix is intended to serve as \code{sm} argument in the \code{\link{seqdist}} function that computes distances between sequences. \code{seqsubm} is an alias that returns only the substitution cost matrix, i.e., no indel.
 }

\details{
The substitution-cost matrix has dimension \eqn{ns*ns}, where
\eqn{ns} is the number of states in the \link{alphabet} of the
sequence object. The element \eqn{(i,j)} of the matrix is the cost of
substituting state \eqn{i} with state \eqn{j}. It represents the dissimilarity between the states \eqn{i} and \eqn{j}. The indel cost of the cost of inserting or deleting a state.

With method \code{CONSTANT}, the substitution costs are all set equal to the \code{cval} value, the default value being 2.

With method \code{TRATE}
(transition rates), the transition probabilities between all pairs of
states is first computed (using the \link{seqtrate} function). Then, the
substitution cost between states \eqn{i} and \eqn{j} is obtained with
the formula

 \deqn{SC(i,j) = cval - P(i|j) -P(j|i)}

where \eqn{P(i|j)} is the probability of transition from state \eqn{j} to
\eqn{i} \code{lag} positions ahead. Default \code{cval} value is 2. When \code{time.varying=TRUE} and \code{transition="both"}, the substitution cost at position \eqn{t} is set as

 \deqn{SC(i,j,t) = cval - P(i|j,t-1) -P(j|i,t-1) - P(i|j,t) - P(j|i,t)}

where \eqn{P(i|j,t-1)} is the probability to transit from state \eqn{j} at \eqn{t-1} to \eqn{i} at \eqn{t}. Here, the default \code{cval} value is 4.

With method \code{FUTURE}, the cost between \eqn{i} and \eqn{j} is the Chi-squared distance between the vector (\eqn{d(alphabet | i)}) of probabilities of transition from states \eqn{i} and
\eqn{j} to all the states in the alphabet \code{lag} positions ahead:

 \deqn{SC(i,j) = ChiDist(d(alphabet | i), d(alphabet | j))}


With method \code{FEATURES}, each state is characterized by the variables \code{state.features}, and the cost between \eqn{i} and \eqn{j} is computed as the Gower distance between their vectors of \code{state.features} values.

With methods \code{INDELS} and \code{INDELSLOG}, values of indels are first derived from the state relative frequencies \eqn{f_i}. For \code{INDELS}, \eqn{indel_i = 1/f_i} is used, and for \code{INDELSLOG}, \eqn{indel_i = \log[2/(1 + f_i)]}{indel_i = log[2/(1 + f_i)]}.
Substitution costs are then set as \eqn{SC(i,j) = indel_i + indel_j}.

For all methods but \code{INDELS} and \code{INDELSLOG}, the indel is set as \eqn{\max(sm)/2}{max(sm)/2} when \code{time.varying=FALSE} and as \eqn{1} otherwise.
}


\usage{
seqcost(seqdata, method, cval = NULL, with.missing = FALSE, miss.cost = NULL,
  time.varying = FALSE, weighted = TRUE, transition = "both", lag = 1,
  miss.cost.fixed = NULL, state.features = NULL, feature.weights = NULL,
  feature.type = list(), proximities = FALSE)

seqsubm(...)
}

\arguments{
  \item{seqdata}{A sequence object as returned by the \link{seqdef} function.}

  \item{method}{String. How to generate the costs. One of \code{"CONSTANT"} (same cost for all substitutions), \code{"TRATE"} (derived from the observed transition rates), \code{"FUTURE"} (Chi-squared distance between conditional state distributions \code{lag} positions ahead), \code{"FEATURES"} (Gower distance between state features), \code{"INDELS"}, \code{"INDELSLOG"} (based on estimated indel costs).}

  \item{cval}{Scalar. For method \code{"CONSTANT"}, the single substitution cost. \cr
  For method \code{"TRATE"}, a base value from which transition probabilities are subtracted.\cr
  If \code{NULL}, \code{cval=2} is used, unless \code{transition}
  is \code{"both"} and \code{time.varying} is \code{TRUE}, in which case \code{cval=4}.}

  \item{with.missing}{Logical. Should an additional entry be added in the matrix for the missing states?
  If \code{TRUE}, the \sQuote{missing} state is also added to the alphabet. Set as \code{TRUE} if you want to use the costs for distances between sequences containing non deleted (non void) missing values. Forced as \code{FALSE} when there are no non-void missing values in \code{seqdata}. See \cite{Gabadinho et al. (2010)} for more details on the options for handling missing values when creating the state sequence object with \code{\link{seqdef}}.}

    \item{miss.cost}{Scalar or vector. Cost for substituting the missing state. Default is \code{cval}.}

    \item{miss.cost.fixed}{Logical. Should the substitution cost for missing be set as the \code{miss.cost} value. When \code{NULL} (default) it will be set as \code{FALSE} when \code{method = "INDELS"} or \code{"INDELSLOG"}, and \code{TRUE} otherwise.}

    \item{time.varying}{Logical. If \code{TRUE} return an \code{\link{array}} with a distinct matrix for each time unit. Time is the third dimension (subscript) of the returned array. Time varying works only with \code{method='CONSTANT'}, \code{'TRATE'}, \code{'INDELS'}, and \code{'INDELSLOG'}.}

    \item{weighted}{Logical. Should weights in \code{seqdata} be used when applicable?}

    \item{transition}{String. Only used if \code{method="TRATE"} and \code{time.varying=TRUE}. On which transition are rates based? Should be one of \code{"previous"} (from previous state), \code{"next"} (to next state) or \code{"both"}.}

    \item{lag}{Integer. For methods \code{TRATE} and \code{FUTURE} only. Time ahead to which transition rates are computed
    (default is \code{lag=1}).}

    \item{state.features}{Data frame with features values for each state.}
    \item{feature.weights}{Vector of feature weights with a weight per column of \code{state.features}.}
    \item{feature.type}{List of feature types. See \code{\link[cluster]{daisy}} for details.}
    \item{proximities}{Logical: should state proximities be returned instead of substitution costs?}
    \item{...}{Arguments passed to \code{seqcost}}
}

\value{
  For \code{seqcost}, a list of two elements, \code{indel} and \code{sm} or \code{prox}:
    \item{indel}{
      The indel cost. Either a scalar or a vector of size \eqn{ns}. When \code{time.varying=TRUE} and \code{method} is one of \code{"INDELS"} or \code{"INDELSLOG"}, a matrix with indels per time point in columns.
    }
    \item{sm}{
      The substitution-cost matrix (or array) when \code{proximities = FALSE} (default).
    }
    \item{prox}{
      The state proximity matrix when \code{proximities = TRUE}.
    }
    \code{sm} and \code{prox} are, when \code{time.varying = FALSE}, a matrix of size \eqn{ns * ns}, where \eqn{ns}
    is the number of states in the alphabet of the sequence object. When \code{time.varying = TRUE}, they are a three dimensional array of size \eqn{ns * ns * L}, where \eqn{L} is the maximum sequence length.

  For \code{seqsubm}, only one element, the matrix (or array) \code{sm}.
}

\seealso{
 \code{\link{seqtrate}}, \code{\link{seqdef}}, \code{\link{seqdist}}.
}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2010). Mining Sequence Data in
\code{R} with the \code{TraMineR} package: A user's guide. Department of Econometrics and
Laboratory of Demography, University of Geneva.

Studer, M. & Ritschard, G. (2016), "What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures", \emph{Journal of the Royal Statistical Society, Series A}. \bold{179}(2), 481-511. \doi{10.1111/rssa.12125}

Studer, M. and G. Ritschard (2014). "A Comparative Review of Sequence Dissimilarity Measures". \emph{LIVES Working Papers}, \bold{33}. NCCR LIVES, Switzerland, 2014. \doi{10.12682/lives.2296-1658.2014.33}
}

\examples{
## Defining a sequence object with columns 10 to 25
## of a subset of the 'biofam' example data set.
data(biofam)
biofam.seq <- seqdef(biofam[501:600,10:25])

## Indel and substitution costs based on log of inverse state frequencies
lifcost <- seqcost(biofam.seq, method="INDELSLOG")
## Here lifcost$indel is a vector
biofam.om <- seqdist(biofam.seq, method="OM", indel=lifcost$indel, sm=lifcost$sm)

## Optimal matching using transition rates based substitution-cost matrix
## and the associated indel cost
## Here trcost$indel is a scalar
trcost <- seqcost(biofam.seq, method="TRATE")
biofam.om <- seqdist(biofam.seq, method="OM", indel=trcost$indel, sm=trcost$sm)

## Using costs based on FUTURE with a forward lag of 4
fucost <- seqcost(biofam.seq, method="FUTURE", lag=4)
biofam.om <- seqdist(biofam.seq, method="OM", indel=fucost$indel, sm=fucost$sm)

## Optimal matching using a unique substitution cost of 2
## and an insertion/deletion cost of 3
ccost <- seqsubm(biofam.seq, method="CONSTANT", cval=2)
biofam.om.c2 <- seqdist(biofam.seq, method="OM",indel=3, sm=ccost)

## Displaying the distance matrix for the first 10 sequences
biofam.om.c2[1:10,1:10]

## =================================
## Example with weights and missings
## =================================
data(ex1)
ex1.seq <- seqdef(ex1[,1:13], weights=ex1$weights)

## Unweighted
subm <- seqcost(ex1.seq, method="INDELSLOG", with.missing=TRUE, weighted=FALSE)
ex1.om <- seqdist(ex1.seq, method="OM", indel=subm$indel, sm=subm$sm, with.missing=TRUE)

## Weighted
subm.w <- seqcost(ex1.seq, method="INDELSLOG", with.missing=TRUE, weighted=TRUE)
ex1.omw <- seqdist(ex1.seq, method="OM", indel=subm.w$indel, sm=subm.w$sm, with.missing=TRUE)

ex1.om == ex1.omw
}

\author{Gilbert Ritschard and Matthias Studer (and Alexis Gabadinho for first version of \code{seqsubm})}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity measures}
