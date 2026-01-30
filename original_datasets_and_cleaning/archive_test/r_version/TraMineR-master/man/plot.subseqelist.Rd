\name{plot.subseqelist}
\alias{plot.subseqelist}
\title{Plot frequencies of subsequences}
\description{
Plot frequencies of subsequences.
}
\usage{
\method{plot}{subseqelist}(x, freq=NULL,cex=1,...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{The subsequences to plot (a \code{subseqelist} object}
  \item{freq}{The frequencies to plot, support if \code{NULL}}
  \item{cex}{Plotting text and symbols magnification. See \code{\link{par}}.}
  % \item{ylim}{A 2 length vector indicating minimum and maximum y values (by default to c(0,1))}
  \item{\dots}{arguments passed to \code{\link{barplot}}}
}
%\details{}
%\references{ ~put references to the literature/web site here ~ }
%\author{ ~~who you are~~ }
%\note{ ~~further notes~~

% ~Make other sections like Warning with \section{Warning }{....} ~
%}
\seealso{
    \code{\link{seqefsub}}
    }
\examples{
## loading data
data(actcal.tse)

## creating sequences
actcal.eseq <- seqecreate(actcal.tse)

## Looking for frequent subsequences
fsubseq <- seqefsub(actcal.eseq,pmin.support=0.01)

## Frequence of first ten subsequences
plot(fsubseq[1:10], cex=2)
plot(fsubseq[1:10])

}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
\keyword{Method}
