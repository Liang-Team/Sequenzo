\name{print.stslist}
\alias{print.stslist}
\title{Print method for state sequence objects}
\description{
This is the print method for state sequence objects of class \code{stslist} created by the \code{seqdef} function.
}
\usage{
\method{print}{stslist}(x, format='STS', extended=FALSE, ...)
}
\arguments{
  \item{x}{A state sequence (\code{stslist}) object.}

  \item{format}{String: print format. One of \code{"STS"} (default) or \code{"SPS"}.}

  \item{extended}{Logical: should the output be printed in extended matrix form?}
  
  \item{...}{Additional print arguments.}
}

\seealso{\code{\link{seqdef}}, \code{\link{plot.stslist}}
}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam <- biofam[500:600,] ## using a subsample only
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
    "Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.shortlab <- c("P","L","M","LM","C","LC","LMC","D")
biofam.seq <- seqdef(biofam, 10:25, states=biofam.shortlab, 
  labels=biofam.lab)

## Print of first 5 sequences
print(biofam.seq[1:5,])
print(biofam.seq[1:5,], extended=TRUE)
print(biofam.seq[1:5,], format="SPS")
print(biofam.seq[1:5,], format="SPS", SPS.out = list(xfix = "", sdsep = "/"))

}

\author{Gilbert Ritschard}
\keyword{Method}
\keyword{State sequences}
