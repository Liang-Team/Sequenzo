\name{seqdecomp}
\alias{seqdecomp}
\title{Convert a character string into a vector of states or events}
\description{
  States can be represented by any substring that does not include the \code{sep} value. An empty separator \code{sep = ""} can only be used when each state is coded with a single character.
}
\usage{
 seqdecomp(data, var=NULL, sep='-', miss="NA", vnames=NULL)
}
\arguments{
 \item{data}{a dataframe, matrix, or character string vector containing sequence data (tibble will be converted with \code{as.data.frame}).}
  \item{var}{the list of columns containing the sequences. Default is \code{NULL}, ie all the columns. Whether the sequences are in the compressed (character strings) or extended format is automatically detected by counting the number of columns.}
  \item{sep}{the between states/events separator used in the input data set. Default is '\code{-}'.}
  \item{miss}{the symbol for missing values (if any) used in the input data set. Default is \code{NA}.}
  \item{vnames}{optional names for the column/variables of the output data set. Default is \code{NULL}.}
}
\seealso{
 \code{\link{seqconc}}.
}
\examples{
# 1 sequence of length 4
seqdecomp("A-BB-C-DD")

# 2 sequences of length 6
seqdecomp(c("ABBCDD","BBCCAD"),sep="")
}
\keyword{Data handling}
