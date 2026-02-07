\name{famform}
\docType{data}
\alias{famform}
\title{Example data set: sequences of family formation}
\description{
This data set contains 5 sequences of family formation histories, used by Elzinga (2008) to introduce several metrics for computing distances between sequences. These sequences don't contain information about the duration spent in each state, they contain only distinct successive states.
}
\details{
The sequences are in `STS' format and stored in character strings with states separated with `-'. The alphabet is made of the five tokens M, MC, S, SC, and U.

This data set is used in TraMineR's manual to crosscheck some results with those presented by Elzinga.
}
\usage{data(famform)}
\format{A data frame with 5 rows and 1 variable.}
\source{Elzinga (2008)}
\references{
Elzinga, Cees H. (2008). Sequence analysis: Metric representations of categorical time
series. Non published manuscript. VU University, Amsterdam.
}

\examples{
data(famform)
ff.seq <- seqdef(famform)
seqiplot(ff.seq)
}

\author{Gilbert Ritschard and Alexis Gabadinho}
\keyword{Datasets}
