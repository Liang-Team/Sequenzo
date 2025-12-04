\name{seqdiff}
\alias{seqdiff}
\alias{print.seqdiff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Position-wise discrepancy analysis between groups of sequences}
\description{
  The function analyses how the differences between groups of sequences evolve along the positions. It runs a sequence of  discrepancy analyses on sliding windows.
}
\usage{
seqdiff(seqdata, group, cmprange = c(0, 1),
  seqdist.args = list(method = "LCS", norm = "auto"), with.missing = FALSE,
  weighted = TRUE, squared = FALSE, seqdist_arg)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{a state sequence object created with the \code{\link{seqdef}} function.}
  \item{group}{The group variable.}
  \item{cmprange}{Vector of two integers: Time range of the sliding windows. Comparison at \eqn{t} is computed on the window (\eqn{t + }\code{cmprange[1]}, \eqn{t + }\code{cmprange[2]}).}
  \item{seqdist.args}{List of arguments passed to \code{\link{seqdist}} for computing the distances.}
  \item{with.missing}{Logical. If \code{TRUE}, missing values are considered as an additional state. If \code{FALSE} subsequences with missing values are removed from the analysis.}
  \item{weighted}{Logical. If \code{TRUE}, \code{seqdiff} uses the weights specified in \code{seqdata}.}
  \item{squared}{Logical. If \code{TRUE} the dissimilarities are squared for computing the discrepancy.}
  \item{seqdist_arg}{Deprecated. Use \code{seqdist.args} instead.}
}
\details{
  The function analyses how the part of discrepancy explained by the \code{group} variable evolves along the position axis. It runs successively discrepancy analyses within a sliding time-window of range \code{cmprange}). At
  each position \eqn{t}, the method uses \code{\link{seqdist}} to compute a distance matrix over the time-window (\eqn{t + }\code{cmprange[1]}, \eqn{t + }\code{cmprange[2]}) and then derives the explained discrepancy on that window with \code{\link{dissassoc}}.

  There are print and plot methods for the returned value.}

\value{
  A \code{seqdiff} object, with the following items:

  \item{stat}{A \code{data.frame} with five statistics (Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, 
  and Levene) for each time stamp of the sequence
      (see \code{\link{dissassoc}})}

  \item{discrepancy}{A \code{data.frame} with, at each time position \eqn{t}, the discrepancy within the whole set of sequences and within each group (defined by the \code{group} variable).}
}

\references{
  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010)
  Discrepancy analysis of complex objects using dissimilarities.
  In F. Guillet, G. Ritschard, D. A. Zighed and H. Briand (Eds.),
  \emph{Advances in Knowledge Discovery and Management},
  Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2009)
  Analyse de dissimilarités par arbre d'induction. In EGC 2009,
  \emph{Revue des Nouvelles Technologies de l'Information}, Vol. E-15, pp. 7-18.

 }
\seealso{\code{\link{dissassoc}} to analyse the association of the \code{group} variable with the whole sequence}
\examples{
## Define a state sequence object
data(mvad)
## First 12 months of first 100 trajectories
mvad.seq <- seqdef(mvad[1:100, 17:28])

## Position-wise discrepancy analysis using
##  centered sliding windows of length 5.
mvad.diff <- seqdiff(mvad.seq, group=mvad$gcse5eq[1:100], cmprange=c(-2,2))
print(mvad.diff)
plot(mvad.diff, stat=c("Pseudo R2", "Levene"))
plot(mvad.diff, stat="discrepancy")
}

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
%\keywords{States Sequences}
