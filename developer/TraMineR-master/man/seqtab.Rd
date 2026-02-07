\name{seqtab}
\alias{seqtab}
\alias{print.stslist.freq}
\title{Frequency table of the sequences}
\description{
Computes the frequency table of the sequences (count and percent of each sequence).
}
\usage{
seqtab(seqdata, idxs = 1:10, weighted = TRUE, format = "SPS", tlim)
}
\arguments{
  \item{seqdata}{a sequence object as defined by the \code{\link{seqdef}} function.}
  \item{idxs}{returns the table for the sequences at ranks '\code{idxs}' in the list of distinct sequences sorted in decreasing order of their frequencies. Default is \code{1:10}, i.e. the 10 most frequent sequences. Can be any subset, like \code{5:10} (fifth to tenth most frequent sequences) or \code{c(2,10)} (second and tenth most frequent sequences). Set \code{idxs=0} to get the table for the whole set of distinct sequences.
}
  \item{weighted}{if \code{TRUE} (default), frequencies account for the weights, if any, assigned to the state sequence object
  (see \code{\link{seqdef}}). Set to \code{FALSE} for ignoring weights.}
  \item{format}{format used for displaying the rownames (the sequences) in the output table.
  Default is SPS format, which yields shorter and more readable sequence representations.
  Alternatively, \code{"STS"} may be specified.}
  \item{tlim}{Deprecated. Use \code{idxs} instead.}
}

\details{The \code{weighted} argument has no effect when no weights were assigned to the state sequence object since weights default in that case to 1.}

\value{An object of class \code{stslist.freq}. This is actually a state sequence object (containing a list of state sequences) with added attributes, among others the \code{freq} attribute containing the frequency table. There are \code{print} and \code{plot} methods for such objects. More sophisticated plots can be produced with the \code{seqplot} function.}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.
}

\seealso{
 \code{\link{seqplot}, \link{plot.stslist.freq}}.
}
\examples{
## Creating a sequence object from the actcal data set
data(actcal)
actcal.lab <- c("> 37 hours", "19-36 hours", "1-18 hours", "no work")
actcal.seq <- seqdef(actcal, 13:24, labels=actcal.lab)

## 10 most frequent sequences in the data
seqtab(actcal.seq)

## With idxs=0, we get all distinct sequences in the data set
## sorted in decreasing order of their frequency
stab <- seqtab(actcal.seq, idxs=0)
head(stab)
tail(stab)

## Example with weights
## from biofam data set using weigths
data(ex1)
ex1.seq <-  seqdef(ex1, 1:13, weights=ex1$weights)

## Unweighted frequencies
seqtab(ex1.seq, weighted=FALSE)

## Weighted frequencies
seqtab(ex1.seq, weighted=TRUE)
}

\author{Alexis Gabadinho (with Gilbert Ritschard for the help page)}

\keyword{Transversal characteristics}

