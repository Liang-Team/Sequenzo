\name{seqfposend}
\alias{seqfposend}
\title{End of first spell in given state}
\description{Returns the position in the sequences of end of first spell in a given state
}

\usage{
seqfposend(seqdata, state, with.missing=FALSE, lead=0, from.seq.start=TRUE)
}

\arguments{
    \item{seqdata}{State sequence object of class \code{stslist} as produced by \code{\link{seqdef}}.}
    \item{state}{Element of the alphabet of \code{seqdata}.}
    \item{with.missing}{Logical. Should non-void missing values be considered as regular states? See \code{\link{seqdss}} and \code{\link{seqdur}}.}
    \item{lead}{Integer. Value to be added to the end position.}
    \item{from.seq.start}{Logical. Should position be computed from the start of the sequence? Default is \code{TRUE}. If \code{FALSE}, position is computed from the start of the spell.}
}

\value{
    Vector of integers giving position of end of spell in the sequences.
}

\seealso{
    \code{\link{seqfpos}}
}

\author{Gilbert Ritschard}

\examples{
## End of spell in further education (FE) in first 10 mvad sequences

data(mvad)
m.seq <- seqdef(mvad[1:10,17:86])
seqfposend(m.seq, state="FE")

}

