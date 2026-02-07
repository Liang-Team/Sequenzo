\name{seqdss}
\alias{seqdss}
\title{Extract sequences of distinct successive states}
\description{
Extract the sequence of distinct successive states from each sequence in a  object.
}
\details{
Returns a sequence object containing the sequences of distinct successive states (DSS). The spell durations are not taken into account. E.g., the DSS contained in \code{'D-D-D-D-A-A-A-A-A-A-A-D'} is \code{'D-A-D'}. Associated durations can be extracted with the \code{\link{seqdur}} function.

When \code{{with.missing=TRUE}}, non-void missing values are considered as a regular state of the alphabet. For example, the DSS of \code{A-A-*-*-*-B-B-C-C-D} is \code{A-*-B-C-D}.

When \code{with.missing=FALSE} (default) missing values are ignored and a substring \code{A-A-*-*-*A} for example will be considered as a single spell in \code{A} while the DSS of this substring would be \code{A-*-A} whith \code{with.missing=TRUE}.

See \link{seqdef} on options for handling missing values when creating sequence objects.

}
\usage{
seqdss(seqdata, with.missing=FALSE)
}
\arguments{
   \item{seqdata}{a sequence object as defined by the \code{\link{seqdef}} function.}
	\item{with.missing}{Should non-void missing values be considered as regular states? See Details.}
}
\value{a sequence object containing the distinct state sequence (DSS) for each sequence in the object given as argument. }
\seealso{
 \code{\link{seqdur}}.
}
\examples{
## Creating a sequence object from columns 13 to 24
## in the 'actcal' example data set
## Here we retain the first 10 sequences only.
data(actcal)
actcal.seq <- seqdef(actcal[1:10,13:24])

## Retrieving the DSS
actcal.dss <- seqdss(actcal.seq)

## Displaying the DSS for the first 10 sequences
actcal.dss

## Example with with.missing argument
data(ex1)
ex1.seq <- seqdef(ex1[, 1:13])

seqdss(ex1.seq)
seqdss(ex1.seq, with.missing=TRUE)
}
\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Longitudinal characteristics}
