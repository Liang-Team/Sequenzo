\name{seqgranularity}
\alias{seqgranularity}
\title{
  Changing sequence time granularity by aggregating positions
}
\description{
Changes time granularity of a state sequence object by aggregating successive positions into groups of a user-defined time length.
}
\usage{
seqgranularity(seqdata, tspan = 3, method = "last")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{
    A state sequence object.
}
  \item{tspan}{
    Integer. Number of successive positions grouped together.
}
  \item{method}{
    Character string. Aggregating method. One of \code{"first"}, \code{"first.valid"}, \code{"last"} (default), \code{"last.valid"}, \ or \code{"mostfreq"}.
}
}
\details{
Successive positions are aggregated by group of \code{tspan} states. The aggregated state is, depending of the \code{method} chosen, either the first (\code{"first"}), the first valid (\code{"first.valid"}), the last (\code{"last"}), the last valid (\code{"last.valid"}), or the most frequent (\code{"mostfreq"}) state of the \code{tspan} long spell. The same applies to the last spell, even when it is shorter than \code{tspan}.

Methods (\code{"first"}) and (\code{"last"}) differ from (\code{"first.valid"}) and (\code{"last.valid"}) only when sequences contain missing values and/or have different lengths.

When there are (void or non void) missings, method \code{"mostfreq"} replaces each interval with the most frequent valid state on the interval or the missing state when there are no valid state.

End missings are set as void when there are voids in \code{seqdata} and as the \code{nr} attribute otherwise.
}
\value{
A \code{stslist} object: The compacted state sequence object.
}
\author{
Matthias Studer and Gilbert Ritschard
}

%% ~Make other sections like Warning with
%\section{Warning }{This function needs further testing.}

\seealso{
 \code{\link[TraMineR]{seqdef}}
}
\examples{
data(mvad)
mvad <- mvad[1:100,]
mvad.seq <- seqdef(mvad[,17:86], xtstep=12)
mvadg.seq <- seqgranularity(mvad.seq, tspan=6, method="first")
par(mfrow=c(2,1))
seqdplot(mvad.seq, with.legend=FALSE, border=NA)
seqdplot(mvadg.seq, with.legend=FALSE)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{utility}% __ONLY ONE__ keyword per line
