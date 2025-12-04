\name{seqlogp}
%\Rdversion{1.1}
\alias{seqlogp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Logarithm of the probabilities of state sequences
}
\description{
Logarithm of the probabilities of state sequences. The probability of a sequence is defined as the product of the probabilities of the successive states in the sequence. State probabilities can either be provided or be computed with one of a few basic models.
}
\usage{
seqlogp(seqdata, prob="trate", time.varying=TRUE,
        begin="freq", weighted=TRUE, with.missing=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
	\item{seqdata}{A state sequence object as produced by \code{\link{seqdef}}.}
	
	\item{prob}{String or numeric array. If a string, either \code{"trate"} or \code{"freq"} to select a probability model to compute the state probabilities. If a numeric array, a matrix or 3-dimensional array of transition probabilities. See details.}
	
	\item{time.varying}{Logical. If \code{TRUE}, the probabilities (transitions or frequencies) are computed separately for each time \eqn{t} point.}
	
	\item{begin}{String of numeric vector. Distribution used to determine the probability of the first state. If a vector, the probabilites to use. If a string, either \code{"freq"} or \code{global.freq}. With \code{freq}, the observed distribution at first position is used. If \code{global.freq}, the overall distribution is used. Default is \code{"freq"}.}
	
	\item{weighted}{Logical. Should we account for the weights when present in \code{seqdata}? Default is \code{TRUE}.}

  \item{with.missing}{Logical. Should non void missing states be treated as regular values? Default is \code{FALSE}.}

}
\details{
The sequence likelihood \eqn{P(s)} is defined as the product of the probability with which each of its observed successive state is supposed to occur at its position.
Let \eqn{s=s_{1}s_{2} \cdots s_{\ell}}{s=s_1s_2 ... s_l} be a sequence of length \eqn{\ell}{l}. Then
\deqn{
	P(s)=P(s_{1},1) \cdot P(s_{2},2) \cdots P(s_{\ell},\ell)
}{
	P(s)=P(s_1, 1) * P(s_2, 2) * ... * P(s_l, l)
}
with \eqn{P(s_{t},t)}{P(s_t,t)} the probability to observe state \eqn{s_t} at position \eqn{t}.

There are different ways to determine the state probabilities \eqn{P(s_t,t)}. The method is chosen by means of the \code{prob} argument.

With \code{prop = "freq"}, the probability \eqn{P(s_{t},t)}{P(s_t,t)} is set as the observed relative frequency at position \eqn{t}. In that case, the probability does not depend on the probabilities of transition. By default (\code{time.varying=TRUE}), the relative frequencies are computed separately for each position \eqn{t}. With \code{time.varying=FALSE}, the relative frequencies are computed over the entire covered period, i.e. the same frequencies are used at each \eqn{t}.


Option \code{prop = "trate"} assumes that each \eqn{P(s_t,t)}, \eqn{t>1} is set as the transition probability \eqn{p(s_t|s_{t-1})}{p(s_t|s_(t-1))}. The state distribution used to determine the probability of the first state \eqn{s_1} is set by means of the \code{begin} argument (see below). With the default \code{time.varying=TRUE}), the transition probabilities are estimated separately at each position, yielding an array of transition matrices. With \code{time.varying=FALSE}, the transition probabilities are assumed to be constant over the successive positions and  are estimated over the entire sequence duration, i.e. from all observed transitions.

Custom transition probabilities can be provided by passing a matrix or a 3-dimensional array as \code{prob} argument.

The distribution used at the first position is set by means of the \code{begin} argument. You can either pass the distribution (probabilities of the states in the alphabet including the missing value when \code{with.missing=TRUE}), or specify \code{"freq"} for the observed distribution at the first position, or \code{global.freq} for the overall state distribution.


The likelihood \eqn{P(s)} being generally very small, \code{seqlogp} returns \eqn{-\log P(s)}{-log(P(s))}. The latter quantity is minimal when \eqn{P(s)} is equal to \eqn{1}.
}
\value{
	Vector of the negative logarithm \eqn{-\log P(s)}{-log(P(s))} of the sequence probabilities.
}

\examples{
## Creating the sequence objects using weigths
data(biofam)
biofam.seq <-  seqdef(biofam, 10:25, weights=biofam$wp00tbgs)

## Computing sequence probabilities
biofam.prob <- seqlogp(biofam.seq)
## Comparing the probability of each cohort
cohort <- biofam$birthyr>1940
boxplot(biofam.prob~cohort)

}

\author{Matthias Studer, Alexis Gabadinho, and Gilbert Ritschard}

\keyword{State sequences}
\keyword{Longitudinal characteristics}  