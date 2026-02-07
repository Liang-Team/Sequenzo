\encoding{utf8}

\name{seqprecstart}
\alias{seqprecstart}

\title{State precarity}

\description{
Determines the state undesirableness degree from the state ordering or conversely the state undesirableness values from the state ordering. (See details.)
}

\usage{
seqprecstart(seqdata,
    state.order=alphabet(seqdata, with.missing), state.equiv=NULL,
    stprec=NULL, with.missing=FALSE)
}

\arguments{
  \item{seqdata}{a state sequence object (class \code{stslist}) as returned by the \code{\link[TraMineR]{seqdef}} function.}

  \item{state.order}{Vector of short state labels defining the order of the states. First the most desirable (most positive) state and then the other states in decreasing desirability order. States of the alphabet that are not included here define the non-comparable states.}

  \item{state.equiv}{List of state equivalence classes. The classes in the list are each set as the vector of the short labels of the states forming the class. }

  \item{stprec}{Vector of user defined state undesirableness degrees ordered conformably with the alphabet of \code{seqdata}. Use negative values to indicate non-comparable states.}

  \item{with.missing}{Logical. Should the missing state be considered as an element of the alphabet?
      }

}


\details{
When \code{stprec} is provided, the order is determined from the \code{stprec} values and overwrites \code{state.order}.

When \code{stprec=NULL}, the initial state undesirableness degrees are set as equivalently spaced values between 0 and 1 assigned to the states in specified order by \code{state.order}.

Equivalent states get the mean value of the states in the equivalence class.

Incomparable states (those not on the \code{state.order} list and not member of an equivalent class having a state listed in \code{state.order}) receive the average of all state undesirableness degrees.

When \code{stprec} is user defined, the provided vector is normalized into a vector with minimum 0 and maximum 1 and mean value of states in a class are assigned to all class members.

Only one state per class needs to be on the state order list. If more than one member is on the state order list they should be listed consecutively. Note that currently no check is performed.

When \code{with.missing=TRUE}, use the \code{seqdata, "nr"} argument to possibly include it in the \code{state.order} or \code{state.equiv}.

}



\value{

The vector of assigned undesirableness degrees sorted according to the original order of states in the alphabet.


}

\seealso{
 \code{\link{seqprecarity}}.
}

\references{

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.

Ritschard, G., Bussi, M., and O'Reilly, J. (2018), "An index of precarity for
measuring early employment insecurity", in G. Ritschard, and M. Studer, \emph{Sequence Analysis and Related Approaches: Innovative Methods and Applications}, Series Life Course Research and Social
Policies, Vol. 10, pp 279-295. Cham: Springer.


}

\examples{
## Defining a sequence object with columns 13 to 24
## in the 'actcal' example data set
data(actcal)
actcal <- actcal[1:200,] ## Here, only a subset
actcal.seq <- seqdef(actcal[,13:24])

## state precarity levels using the original state order
seqprecstart(actcal.seq)

## Assuming A and B as equivalent regarding precarity
seqprecstart(actcal.seq, state.equiv=list(c('A','B')))

## and letting C be non-comparable
seqprecstart(actcal.seq, state.order=c("A","B","D"), state.equiv=list(c('A','B')))

}

\author{Gilbert Ritschard}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Sequence analysis}
