\name{seqauto}
\alias{seqauto}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Auto-association between states
}
\description{
 Computes auto-associations of order k = 1 to \code{order}, between current states and states lagged by k positions.
}

\usage{
seqauto(seqdata, order = 1, measure = "cv")
}

\arguments{
  \item{seqdata}{
        A state sequence object or a data frame with sequential data in STS format.
}
  \item{order}{
        Maximum wanted order of auto-association.
}
  \item{measure}{
        Character string. Currently only \code{"cv"} (Cramer's v) is accepted.
}
}

\details{
The function puts the data in \code{"SRS"} form by means of the \code{\link[TraMineR]{seqformat}} function.}

\value{
A matrix with \code{order} rows and  two columns: the auto-association and its p-value.
}
%\references{
%% ~put references to the literature/web site here ~
%}
\author{
    Gilbert Ritschard
}
%\note{
%%%  ~~further notes~~
%}

\section{Warning }{Function in development, not fully checked.}

\seealso{
    \code{\link[TraMineR]{seqformat}}
}
\examples{
data(biofam)

biofam.seq <- seqdef(biofam[1:100,10:25])
aa <- seqauto(biofam.seq, order=5)
aa
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{state sequences}
