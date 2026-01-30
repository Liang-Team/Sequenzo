\name{is.stslist}
\alias{is.stslist}
\title{Test if is a proper state sequence (stslist) object}
\description{The function tests whether \code{x} is of class \code{stslist} and if its \code{weights} attribute has the expected length and names.
}

\usage{
is.stslist(x)
}
\arguments{
 \item{x}{object to be tested.}
 }
\value{
 Logical: result of the test.
}
\seealso{
 \code{\link{seqdef}}
}
\examples{
## Creating a sequence object with the columns 13 to 24
## in the 'actcal' example data set
data(biofam)
biofam <- biofam[sample(nrow(biofam),300),]
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
                "Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam[,10:25], weights=biofam$wp00tbgs)

is.stslist(biofam.seq) #TRUE

attr(biofam.seq,"weights") <- NULL
is.stslist(biofam.seq) #TRUE

attr(biofam.seq,"weights") <- rep(1, nrow(biofam.seq))
is.stslist(biofam.seq) #FALSE

w <- rep(1, nrow(biofam.seq))
names(w) <- rownames(biofam.seq)
attr(biofam.seq,"weights") <- w
is.stslist(biofam.seq) #TRUE

}

\author{Gilbert Ritschard}

\keyword{State sequences}
