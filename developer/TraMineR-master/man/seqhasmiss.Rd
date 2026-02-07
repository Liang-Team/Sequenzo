\name{seqhasmiss}
\alias{seqhasmiss}
\title{Count and identification of sequences with nr and void missings}
\description{The function prints the counts of sequences with included (nr) and ignored (void) missings. It returns invisibly three logical vectors with \code{TRUE} for sequences with nr or void missing (\code{has.miss}), included missing (\code{has.nr}), and ignored missing (\code{has.void}) respectively.
}

\usage{
seqhasmiss(seqdata)
}
\arguments{
 \item{seqdata}{State sequence object of class \code{stslist}.}
}
\value{
 List of three logical indicator vectors: \code{has.miss}, \code{has.nr}, and \code{has.void}.
}
\examples{
data(ex1)
s <- seqdef(ex1[,1:12])
res <- seqhasmiss(s)
res$has.nr
}

\author{Gilbert Ritschard}

\keyword{State sequences}
