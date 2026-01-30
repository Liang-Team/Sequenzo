\name{seqmodst}
\alias{seqmodst}
\title{Sequence of modal states}
\description{
Sequence made of the modal state at each position.
}
\usage{
seqmodst(seqdata, weighted=TRUE, with.missing=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{a state sequence object as defined by the \code{\link{seqdef}} function.}

  \item{weighted}{if \code{TRUE}, distributions account for the weights assigned to the state sequence object
  (see \code{\link{seqdef}}).
  Set as \code{FALSE} if you want ignore the weights.}

  \item{with.missing}{If \code{FALSE} (default value), returned distributions ignore missing values.}
}
\details{
In case of multiple modal states at a given position, the first one is taken. Hence, the result may vary with the alphabet order.
}
\value{
an object of class \emph{stslist.modst}. This is actually a state sequence object (containing a single state sequence) with additional attributes, among which the \code{Frequencies} attribute containing the transversal frequency of each state in the sequence. There are print and plot methods for such objects. More sophisticated plots can be produced with the \code{seqplot} function.
}

\seealso{\code{\link{plot.stslist.modst}} for default plot method, \code{\link{seqplot}} for higher level plots.}


\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.
}

\author{Alexis Gabadinho}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam, 10:25, labels=biofam.lab)

## Modal state sequence
seqmodst(biofam.seq)

## Examples using weights and with.missing arguments
data(ex1)
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)

seqmodst(ex1.seq)
seqmodst(ex1.seq, weighted=FALSE)
seqmodst(ex1.seq, weighted=FALSE, with.missing=TRUE)
}


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Transversal characteristics}
 
