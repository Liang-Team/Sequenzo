\name{stlab}
\alias{stlab}
\alias{stlab<-}
\title{Get or set the state labels of a sequence object}
\description{
This function gets or sets the state labels of a sequence object, that is, the long labels used when displaying the state legend in plotting functions.
}
\details{The state legend is plotted either automatically by the plot functions provided for visualizing sequence objects or with the \code{\link{seqlegend}} function. A long label is associated to each state of the alphabet and displayed in the legend. The state labels are defined when creating the sequence object, either automatically using the values found in the data or by specifying a user defined vector of labels. The \code{stlab} function can be used to get or set the state labels of a previously defined sequence object.}
\usage{
stlab(seqdata)
stlab(seqdata) <- value
}
\arguments{
  \item{seqdata}{a state sequence object as defined by the \code{\link{seqdef}} function.}
  \item{value}{a vector of character strings containing the labels, of length equal to the number of states in the alphabet. Each string is attributed to the corresponding state in the alphabet, the order being the one returned by the \code{\link{alphabet}}.}
}
\value{For 'stlab' a vector containing the labels.

For 'stlab<-' the updated sequence object.
}
\seealso{
 \code{\link{seqdef}}
}
\examples{
## Creating a sequence object with the columns 13 to 24
## in the 'actcal' example data set
## The color palette is automatically set
data(actcal)
actcal.seq <- seqdef(actcal,13:24)

## Retrieving the color palette
stlab(actcal.seq)
seqiplot(actcal.seq)

## Changing the state labels
stlab(actcal.seq) <- c("Full time","Part time (19-36 hours)",
    "Part time (1-18 hours)", "No work")
seqiplot(actcal.seq)
}
\keyword{Sequence-object attributes}
