\name{TraMineR-package}
\alias{TraMineR-package}
\alias{TraMineR}
\docType{package}
\title{
Trajectory Miner: Sequence Analysis Toolkit
}
\description{
(Version: 2.2-12) Set of sequence analysis tools for manipulating,
describing and rendering categorical sequences, and more generally
mining sequence data in the field of social sciences. Although
this sequence analysis package is primarily intended for state or
event sequences that describe time use or life courses such as
family formation histories or professional careers, its features
also apply to many other kinds of categorical sequence data.
It accepts many different sequence representations as input and
provides tools for converting sequences from one format to another.
It offers several functions for describing and rendering sequences,
for computing distances between sequences with different metrics
(among which optimal matching), original dissimilarity-based
analysis tools, and functions for extracting the most frequent
event subsequences and identifying the most discriminating ones
among them. A user's guide can be found on the TraMineR web page.
}

\details{
TraMineR provides tools for both state sequences and event
sequences. The first step when using the package is to define a
state sequence object (with \code{\link{seqdef}}) if you want to
explore state sequences, and an event sequence object (with
\code{\link{seqecreate}}) if you are interested in event
sequencing.

State sequences are defined from a series of variables giving the
states at the successive positions, while event sequences are
defined from (vertical) time stamped event data. The package,
however, can handle many other different data organizations and
provides tools to help converting state sequences into event
sequences and vice versa.
}

\author{
Alexis Gabadinho, Matthias Studer, Nicolas S. Müller, Reto Bürgin, Pierre-Alexandre Fonta, and Gilbert Ritschard

}

\references{
    Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011).
    Analyzing and Visualizing State Sequences in R with TraMineR.
    \emph{Journal of Statistical Software} \bold{40}(4), 1-37.
    \doi{10.18637/jss.v040.i04}.

    Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009).
    Mining Sequence Data in \code{R} with the \code{TraMineR} package:
    A user's guide. Department of Econometrics and Laboratory of
    Demography, University of Geneva
}

\examples{
## load the mvad data
library(TraMineR)
data(mvad)

## create a state sequence object from columns 17 to 86
mvad.seq <- seqdef(mvad[,17:86])

## distribution plot by sex (male)
seqdplot(mvad.seq, group=mvad$male, border=NA)

## compute the LCS pairwise distance matrix
## among the first 10 sequences
mvad.lcs <- seqdist(mvad.seq[1:10,], method="LCS")
}

\keyword{ package }
