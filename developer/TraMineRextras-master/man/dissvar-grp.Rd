\encoding{latin1}
\name{dissvar.grp}
\alias{dissvar.grp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
 Discrepancy by group.
}
\description{
This function computes the dissimilarity-based discrepancy measure of the groups defined by the group variable. The function is a wrapper for the TraMineR \code{\link[TraMineR]{dissvar}} function.
}
\usage{
dissvar.grp(diss, group=NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{
        a dissimilarity matrix or a \code{dist} object.
        }
  \item{group}{
        group variable. If \code{NULL} a single group is assumed.
        }
  \item{...}{
        additional arguments passed to \code{\link[TraMineR]{dissvar}} .
        }
}

\details{
The function is a wrapper for running \code{\link[TraMineR]{dissvar}} on the different groups defined by the \code{group} variable.
}
\value{
A vector with the group discrepancy measures.
}

\note{This function is a pre-release and further testing is still needed, please report any problems.
}
\author{
Gilbert Ritschard
}

\seealso{
\code{\link[TraMineR]{dissvar}}
}
\examples{
## create the biofam.seq state sequence object from the biofam data.
data(biofam)
biofam <- biofam[1:100,]
biofam.seq <- seqdef(biofam[,10:25])
dist <- seqdist(biofam.seq, method="HAM")

## discrepancy based on non-squared dissimilarities
dissvar.grp(dist, biofam$plingu02)
## square root of discrepancy based on squared dissimilarities
sqrt(dissvar.grp(dist, biofam$plingu02, squared=TRUE))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{state sequences}

