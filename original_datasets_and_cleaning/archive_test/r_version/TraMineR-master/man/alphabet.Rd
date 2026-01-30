\name{alphabet}
\alias{alphabet}
\alias{alphabet<-}
\title{Get or set the alphabet of a state or event sequence object}
\description{
For state sequences, the function gets or sets the (short) labels associated to the states in the alphabet of a state sequence object (the list of all possible states). The get form also applies to event sequences, while the set form does not work with event sequences.
}
\details{A state sequence object---created with the \code{\link{seqdef}} function---stores sequences as a matrix where columns are factors. The levels of the factors include the alphabet plus the codes for missing values and void elements. The alphabet function retrieves or sets the \dQuote{alphabet} attribute of the state sequence object.
The state names composing the alphabet are preferably short labels, since they are used for printing sequences. Longer labels for describing more precisely each state in legend are stored in the \dQuote{labels} attribute of the sequence object.

For an event sequence object---created with \code{\link{seqecreate}}---the get form of \code{alphabet} works as an alias for \code{levels}. The set form \code{alphabet <-} does not work and should not be used.
}
\usage{
alphabet(seqdata, with.missing=FALSE)
alphabet(seqdata) <- value
}
\arguments{
 \item{seqdata}{a state sequence object of class \code{stslist} as defined with the \code{\link{seqdef}} function or, for the get form only, an event sequence object as defined with \code{\link{seqecreate}} or a probabilistic suffix tree generated with the PST package.}
 \item{value}{For state sequences only. Vector of characters of the same length as the vector returned by the \code{alphabet} function, i.e. one label for each state in the alphabet.}
 \item{with.missing}{Logical. When \code{seqdata} is a state sequence object (\code{stslist}), should the returned alphabet include the \code{nr} symbol standing for missing states?}
 }
\value{For \sQuote{\code{alphabet}} a character vector containing the alphabet.

For \sQuote{\code{alphabet <-}} the updated state sequence object.
}
\seealso{
 \code{\link{seqdef}}
}
\examples{
## Creating a sequence object with the columns 13 to 24
## in the 'actcal' example data set
data(actcal)
actcal.seq <- seqdef(actcal,13:24)

## Retrieving the alphabet
alphabet(actcal.seq)

## Setting the alphabet
alphabet(actcal.seq) <- c("FT", "PT", "LT", "NO")

## Event sequences
actcal.eseq <- seqecreate(actcal.seq)
alphabet(actcal.eseq)
}

\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Sequence-object attributes}
