\name{FCE_to_TSE}
\alias{FCE_to_TSE}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Data conversion from Fixed Column Event format to TSE.
}
\description{
Data conversion from Fixed Column Event format to TSE.
}
\usage{
FCE_to_TSE(seqdata, id = NULL, cols, eventlist = NULL, firstEvent = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{
	data frame or matrix containing event sequence data in FCE format.
}
  \item{id}{
	column containing the identification numbers for the sequences.
}
  \item{cols}{ Real. Column containing the timing of the event. A missing value is interpreted as a non-occurrence of the event.}
  \item{eventlist}{Event names, specified in the same order as \code{cols} argument. If \code{NULL} (default), column names are used.}
  \item{firstEvent}{Character. The name of an event to be added at the beginning of each event sequences. This allows to include individuals with no events. If \code{NULL} (default), no event is added.}
}
\details{
The usual data format for event sequence is TSE (see \code{\link[TraMineR]{seqecreate}}).
}
\value{
A \code{data.frame} with three columns: "id", "timestamp" and "event".
}
\author{
Matthias Studer
}

\note{This function is a pre-release and further testing is still needed, please report any problems.}


\seealso{
\code{\link[TraMineR]{seqecreate}}, \code{\link[TraMineR]{seqformat}}
}
\examples{
## Generate a ramdom data set
fce <- data.frame(id=1:100, event1=runif(100), event2=runif(100))

## Add missing values (ie non-occurrences)
fce[runif(100)<0.1, "event1"] <- NA
fce[runif(100)<0.1, "event2"] <- NA

tse <- FCE_to_TSE(fce, id="id", cols=c("event1", "event2"),
       eventlist=c("Marriage", "Child birth"), firstEvent="Birth")

seq <- seqecreate(tse)
print(seq[1:10])

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{data format}
