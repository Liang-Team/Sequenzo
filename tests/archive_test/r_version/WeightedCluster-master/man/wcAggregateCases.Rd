\name{wcAggregateCases}
\alias{wcAggregateCases}
\alias{wcAggregateCases.matrix}
\alias{wcAggregateCases.data.frame}
\alias{wcAggregateCases.stslist}
\alias{print.wcAggregateCases}
\title{
Aggregate identical cases.
}
\description{
Function to aggregate identical cases.
}
\usage{
wcAggregateCases(x, weights = NULL, ...)
\method{wcAggregateCases}{data.frame}(x, weights=NULL, ...) 
\method{wcAggregateCases}{matrix}(x, weights=NULL, ...) 
\method{wcAggregateCases}{stslist}(x, weights=NULL, weighted=TRUE, ...)
\method{print}{wcAggregateCases}(x,  ...) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
	\item{x}{The object to aggregate.}
	\item{weights}{Numeric. An optional case weights vector.}
	\item{weighted}{Logical. If \code{TRUE}, the \code{weights} are taken from the sequence object (see \code{\link[TraMineR]{seqdef}}).}
	\item{\dots}{Optional additionnal arguments.}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
A \code{wcAggregateCases} object with the following components:
\describe{
  \item{aggIndex}{Index of the unique cases in the original object data.}
  \item{aggWeights}{Aggregated case weights}
  \item{disaggIndex}{Index of the original object data in the unique cases.}
  \item{disaggWeights}{Original weights used.}
 }
}

\examples{
data(mvad)
## Taking only the father unemployment and 
## success at the end of compulsory schooling.
myData <- mvad[ , c("funemp", "gcse5eq")]
## Computing aggregated cases informations
ac <- wcAggregateCases(myData, weights=mvad$weight)
print(ac)
## Retrieving unique cases in the original data set
uniqueData <- myData[ac$aggIndex, ]
## Table from original data
table.orig <- xtabs(mvad$weight~funemp+gcse5eq, data=myData)

## Table from aggregated data
table.agg <- xtabs(ac$aggWeights~funemp+gcse5eq, data=uniqueData)

## Both table are equal, no information is lost 
## (only the call command is different)
all(table.orig == table.agg)

}
