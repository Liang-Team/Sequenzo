\encoding{latin1}
\name{rowmode}
\alias{rowmode}

\author{Gilbert Ritschard}

\title{Modal state of a variable}


\description{
Returns the modal state of a variable, e.g., the modal state in a sequence.}

\usage{
rowmode(v, except = NULL)
}

\arguments{
  \item{v}{A numerical or factor variable.}

  \item{except}{Vector of values that should be ignored; e.g., set \code{except="*"} to ignore missing states with default coding.}
}

\details{
    The function tabulates the variable and returns the most frequent value.
}

\value{
 The modal value
}

\seealso{
\code{\link{table}}.
}

\examples{
data(actcal)
actcal.seq <- seqdef(actcal[1:10,13:24])
actcal.mod <- apply(as.matrix(actcal.seq), 1, rowmode)
head(actcal.mod)

}
\keyword{util}
