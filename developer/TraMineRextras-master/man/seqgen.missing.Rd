\name{seqgen.missing}
\alias{seqgen.missing}
\title{
  Generate random missing elements within a state sequence object
}
\description{
 The function assigns missing values (\code{nr} attribute of the object, which is \code{"*"} by default) to randomly selected positions in randomly selected cases.
}
\usage{
seqgen.missing(seqdata, p.cases = 0.1, p.left = 0.2, p.gaps = 0, p.right = 0.3,
          mt.left="nr", mt.gaps="nr", mt.right="nr")
}
\arguments{
  \item{seqdata}{
        A state sequence object.
}
  \item{p.cases}{
        Proportion of cases with missing values.
}
  \item{p.left}{
        Proportion of left missing values.
}
  \item{p.gaps}{
        Proportion of gap missing values.
}
  \item{p.right}{
        Proportion of right missing values.
}
  \item{mt.left}{
        Type of left missing. One of \code{"nr"} (non response state) or \code{"void"}
}
  \item{mt.gaps}{
        Type of gap missing. One of \code{"nr"} (non response state) or \code{"void"}
}
  \item{mt.right}{
        Type of right missing. One of \code{"nr"} (non response state) or \code{"void"}
}
}
\details{
The aim of the function is essentially pedagogical. It may serve to illustrate how results of a sequential analysis may be affected by the presence of random missing states.

States in the sequences are randomly replaced with missing values. For each selected sequence, first, a random proportion between 0 and \code{p.gaps} of gaps are randomly inserted, then a random proportion between 0 and \code{p.left} of positions from the start of the sequence are set as missing, and finally a random proportion between 0 and \code{p.right} of positions from the end of the sequence are set as missing. Left missing values may possibly overlap gaps, and right missing values may overlap gaps and/or right missing values.
}
\value{
The resulting state sequence object.
}
\author{
 Gilbert Ritschard
}

%\note{Warning: This function needs further testing.}

%% ~Make other sections like Warning with
\section{Warning }{This function needs further testing.}

\seealso{
 \code{\link[TraMineR]{seqdef}}
}
\examples{
## create the biofam.seq state sequence object from the biofam data.
data(biofam)
biofam.seq <- seqdef(biofam[1:100,10:25])

## Generate missing states within 60% of the sequences.
biofamm.seq <- seqgen.missing(biofam.seq, p.cases=.6,
      p.left=.4, p.gaps=.2, p.right=.5)

## compare the rendering of the sequences before and after
##   introducing missing states.
par(mfrow=c(2,2))
seqIplot(biofam.seq, sortv="from.end", with.legend=FALSE)
seqIplot(biofamm.seq, sortv="from.end", with.legend=FALSE)
seqdplot(biofam.seq, with.missing=TRUE, border=NA, with.legend=FALSE)
seqdplot(biofamm.seq, with.missing=TRUE, border=NA, with.legend=FALSE)
dev.off()
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{utility}
