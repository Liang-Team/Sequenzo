\name{dissvar}
\alias{dissvar}
\title{Dissimilarity based discrepancy}
\description{
  Compute the discrepancy from the pairwise dissimilarities between objects. The discrepancy is a measure of dispersion of the set of objects.
}
\usage{
dissvar(diss, weights=NULL, squared = FALSE)
}
\arguments{
  \item{diss}{A dissimilarity matrix or a \code{dist} object (see \code{\link{dist}})}
  \item{weights}{optional numerical vector containing weights.}
  \item{squared}{Logical. If \code{TRUE} \code{diss} is squared.}
}
\details{
The discrepancy is an extension of the concept of variance to any kind of objects for which we can compute pairwise dissimilarities.
The discrepancy \eqn{s^2} is defined as:
\deqn{s^2=\frac{1}{2n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}d_{ij}}{s^2=(1/(2n^2)) * sum sum d_ij}

\emph{Mathematical ground}:
In the Euclidean case, the sum of squares can be expressed as:
  \deqn{SS=\sum_{i=1}^{n}(y_i-\bar{y})^2=\frac{1}{2n}\sum_{i=1}^{n}\sum_{j=1}^{n}(y_i-y_j)^2}{SS= sum (y_i - y_mean)^2=(1/(2n)) * sum sum (y_i - y_j)^2}

  The concept of discrepancy generalizes the equation by allowing to replace the  \eqn{(y_i - y_j)^2} term with any  measure of dissimilarity \eqn{d_{ij}}.
}

\value{The discrepancy.}

\references{
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010)
  Discrepancy analysis of complex objects using dissimilarities.
  In F. Guillet, G. Ritschard, D. A. Zighed and H. Briand  (Eds.),
  \emph{Advances in Knowledge Discovery and Management},
  Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2009)
  Analyse de dissimilarités par arbre d'induction. In EGC 2009,
  \emph{Revue des Nouvelles Technologies de l'Information}, Vol. E-15, pp. 7-18.

  Anderson, M. J. (2001) A new method for non-parametric multivariate analysis of variance.
\emph{Austral Ecology} \bold{26}, 32-46.

  Batagelj, V. (1988) Generalized ward and related clustering problems. In H. Bock (Ed.),
  \emph{Classification and related methods of data analysis}, Amsterdam: North-Holland, pp. 67-74.

}
\seealso{
   \code{\link{dissassoc}} to test association between objects represented by their dissimilarities and a covariate.
\cr
\code{\link{disstree}} for an induction tree analyse of objects characterized by a dissimilarity matrix.
\cr
\code{\link{disscenter}} to compute the distance of each object to its group center from pairwise dissimilarities.
   \cr
	 \code{\link{dissmfacw}} to perform multi-factor analysis of variance from pairwise dissimilarities.
}

\examples{
## Defining a state sequence object
data(mvad)
mvad.seq <- seqdef(mvad[, 17:86])

## Building dissimilarities (any dissimilarity measure can be used)
mvad.ham <- seqdist(mvad.seq, method="HAM")

## Pseudo variance of the sequences
print(dissvar(mvad.ham))
}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
