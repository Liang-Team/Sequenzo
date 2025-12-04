\name{seqalign}
\alias{seqalign}
\alias{plot.seqalign}
\alias{print.seqalign}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Computation details about a pairwise alignment}
\description{
The function provides details about a pairwise alignment.
}
\usage{
seqalign(seqdata, indices, indel=1, sm, with.missing = FALSE)

\method{plot}{seqalign}(x, cpal = NULL, missing.color = NULL, ylab = NULL,
  yaxis = TRUE, xaxis = TRUE, ytlab = NULL, ylas = 0, xtlab = NULL,
  cex.axis = 1, cex.plot, ...)

\method{print}{seqalign}(x, digits=3, ...)
}
%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{seqdata}{a state sequence object defined with the \code{\link{seqdef}} function.
}
  \item{indices}{
    a vector of length 2 giving the indexes of the two sequences
}
  \item{indel}{
    indel cost (see \code{\link{seqdist}})
}
  \item{sm}{
    matrix of substitution costs or a method for computing the costs (see \code{\link{seqdist}})
}
  \item{with.missing}{
    logical: Should the missing state be considered as an element of the alphabet?
}
    \item{x}{
    an object of class \code{seqalign}}
    \item{cpal}{color palette}
    \item{missing.color}{color for missing elements}
    \item{ylab}{y label}
    \item{yaxis}{yaxis}
    \item{xaxis}{xaxis}
    \item{ytlab}{ytlab}
    \item{ylas}{ylas}
    \item{xtlab}{xtlab}
    \item{cex.axis}{Axis annotation magnification. See \code{\link{par}}.}
    \item{digits}{number of digits for printed output}
  \item{cex.plot}{Deprecated. Use \code{cex.axis} instead.}
  \item{...}{additional arguments passed to other functions}
}
\details{
There are print and plot methods for \code{seqalign} objects.
}

\value{Object of class \code{seqalign}}

%\references{}

\seealso{\code{\link{seqdist}}}

\examples{
data(biofam)
biofam.seq <- seqdef(biofam, 10:25)
costs <- seqsubm(biofam.seq, method="TRATE")
sa <- seqalign(biofam.seq, 1:2, indel=1, sm=costs)
print(sa)
plot(sa)
sa <- seqalign(biofam.seq, c(1,5), indel=0.5, sm=costs)
print(sa)
plot(sa)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\author{Alexis Gabadinho (plot.seqalign) and Matthias Studer (seqalign) (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
