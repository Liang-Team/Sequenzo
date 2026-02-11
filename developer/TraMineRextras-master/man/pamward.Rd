\encoding{latin1}
\name{pamward}
\alias{pamward}

\author{Gilbert Ritschard}

\title{PAM from k-solution of hierarchical clustering}


\description{
Runs a pam clustering (\code{\link[cluster]{pam}}) from the solution in k groups of a hierarchical clustering (\code{\link[cluster]{agnes}}) .
}

\usage{
pamward(diss, k=3, method="ward", dist)
}

\arguments{
  \item{diss}{Distance matrix or object.}

  \item{k}{Integer. Number of clusters.}

  \item{method}{Method for the hierarchical clustering (see \code{\link[cluster]{agnes}}).}
  
  \item{dist}{Deprecated. Use \code{diss} instead.}
}

\details{
    The function first runs the hierarchical clustering, retrieves the medoids of the solution for the provided \code{k} and uses those medoids as start centers for the \code{pam} partitioning.
}

\value{
An object of class \code{"pam"}. See \code{\link[cluster]{pam.object}} for details.   }

\seealso{
\code{\link[cluster]{agnes}} and \code{\link[cluster]{pam}}.
}

\examples{
library(cluster)
data(actcal)
actcal.seq <- seqdef(actcal[1:200,13:24])
actcal.ham <- seqdist(actcal.seq, method = "HAM")
clust <- pamward(actcal.ham, k = 4)
table(clust$clustering)
}

\keyword{util}
