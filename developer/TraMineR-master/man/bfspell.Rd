\name{bfspell}
\docType{data}
\alias{bfspell}
\alias{bfspell20}
\alias{bfpdata20}

\title{Example data set: First 20 biofam sequences in SPELL form}

\description{First 20 sequences of the \code{\link{biofam}} data set in SPELL form. The data serve to illustrate the use of \code{\link{seqformat}} for converting SPELL data into STS (horizontal) form.
}


\details{
The states are coded with the following short labels\cr
\cr
P = "Parent" \cr
L = "Left" \cr
M = "Married" \cr
LM = "Left+Marr" \cr
C = "Child" \cr
LC = "Left+Child" \cr
LMC = "Left+Marr+Child" \cr
D = "Divorced" \cr
\cr
The data is a SPELL representation of \code{biofam[1:20,10:25]}, corresponding to 20 family life sequences between ages 15 and 30.
}

\usage{
data(bfspell)
}

\format{A data set with two data frames: \code{bfspell20} with one row per spell and \code{bfpdata20} with one row per id. The \code{bfspell20} data frame contains the spell data themselves (4 variables \code{id}, \code{begin}, \code{end}, \code{states}) and \code{bfpdata20} the year when aged 15 (2 variables \code{id}, \code{when15}).
}
\seealso{
\code{\link{biofam}}
}
\author{Gilbert Ritschard and Alexis Gabadinho}

\keyword{Datasets}
