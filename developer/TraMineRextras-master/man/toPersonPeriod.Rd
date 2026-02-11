\name{toPersonPeriod}
\alias{toPersonPeriod}

\author{Matthias Studer}

\title{Converting into person-period format.}


\description{
Converts the STS sequences of a state sequence object into person-period format.}

\usage{
toPersonPeriod(seqdata)
}

\arguments{
  \item{seqdata}{A state sequence object as returned by \code{\link[TraMineR]{seqdef}}.}
}

%\details{
%}

\value{
  A data frame with three columns: \code{id}, \code{state} and \code{timestamp}.
}

\seealso{
\code{\link[TraMineR]{seqformat}} .
}

\examples{
data(mvad)
mvad.labels <- c("employment", "further education", "higher education",
    "joblessness", "school", "training")
mvad.scodes <- c("EM", "FE", "HE", "JL", "SC", "TR")
mvad.seq <- seqdef(mvad, 15:86, states = mvad.scodes, labels = mvad.labels)

mvad2 <- toPersonPeriod(mvad.seq[1:20,])
}
\keyword{util}
