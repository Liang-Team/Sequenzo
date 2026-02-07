\name{seqstatl}
\alias{seqstatl}
\title{List of distinct states or events (alphabet) in a sequence data set.}
\description{
  Returns a list containing distinct states or events found in a data frame or matrix containing sequence data, the alphabet.
}
\usage{
 seqstatl(data, var=NULL, format='STS')
}
\arguments{
  \item{data}{a data frame, matrix, or character string vector containing sequence data (tibble will be converted with \code{as.data.frame}).}
  \item{var}{the list of columns containing the sequences. Default \code{NULL} means all columns. Whether the sequences are in the compressed (character strings) or extended format is automatically detected from the number of columns.}
  \item{format}{the format of the sequence data set. One of \code{"STS"}, \code{"SPS"}, \code{"DSS"}. Default is \code{"STS"}. The \code{seqstatl} function uses the \code{\link{seqformat}} function to translate between formats when necessary.}
}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009). Mining Sequence Data in
\code{R} with the \code{TraMineR} package: A user's guide. Department of Econometrics and
Laboratory of Demography, University of Geneva.
}

\seealso{
  \code{\link{seqformat}}
}
\examples{
data(actcal)
seqstatl(actcal,13:24)
}

\author{Alexis Gabadinho}

\keyword{Data handling}
