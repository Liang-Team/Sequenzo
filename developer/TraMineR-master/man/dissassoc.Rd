\name{dissassoc}
\alias{dissassoc}
\alias{hist.dissassoc}
\alias{print.dissassoc}
\title{Analysis of discrepancy from dissimilarity measures}

\description{
Compute and test the share of discrepancy (defined from a dissimilarity matrix) explained by a categorical variable.
}

\usage{
dissassoc(diss, group, weights=NULL, R=1000,
          weight.permutation="replicate", squared=FALSE)
}

%- maybe also 'usage' for other objects documented here.

\arguments{
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}})}
  \item{group}{A categorical variable. For a numerical variable use \code{\link{dissmfacw}}.}
  \item{weights}{optional numerical vector containing weights.}
  \item{R}{Number of permutations for computing the p-value. If equal to 1, no permutation test is performed.}
  \item{weight.permutation}{Weighted permutation method: \code{"diss"}
  (attach weights to the dissimilarity matrix), \code{"replicate"} (replicate case using \code{weights}), \code{"rounded-replicate"} (replicate case using rounded \code{weights}), \code{"random-sampling"} (random assignment of covariate profiles to the objects using distributions defined by the weights.)}
  \item{squared}{Logical. If \code{TRUE} the dissimilarities \code{diss} are squared.}
}

\details{The \code{dissassoc} function assesses the association
between objects characterized by their dissimilarity matrix and a
discrete covariate. It provides a generalization of the ANOVA
principle to any kind of distance metric. The function returns a pseudo F statistic, 
a pseudo Brown-Forsythe Fbf statistic, and
a pseudo R-square that can be interpreted as a usual R-square. The
statistical significance of the association is computed by means of
permutation tests. The function performs also a test of discrepancy
homogeneity (equality of within variances) using a generalization of
the Levene statistic and the Bartlett statistic. \cr There are
\code{print} and \code{hist} methods (the latter producing an
histogram of the permuted values used for testing the significance).

If a numeric \code{group} variable is provided, it will be treated as categorical, i.e., each different value will be considered as a different category. To measure the `linear' effect of a numerical variable, use \code{\link{dissmfacw}}.
}

\value{ An object of class \code{dissassoc} with the
following components:
  \item{groups}{A data frame with the number of cases and the discrepancy of each group}
  \item{anova.table}{The pseudo ANOVA table}
  \item{stat}{The value of the statistics (Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, and Levene) and their p-values}
  \item{perms}{The permutation object, containing the values computed for each permutation}
}
\references{
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010)
  Discrepancy analysis of complex objects using dissimilarities.
  In F. Guillet, G. Ritschard, H. Briand, and D. A. Zighed (Eds.),
  \emph{Advances in Knowledge Discovery and Management},
  Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2009).
  Analyse de dissimilarités par arbre d'induction. In EGC 2009,
  \emph{Revue des Nouvelles Technologies de l'Information}, Vol. E-15, pp. 7--18.

  Anderson, M. J. (2001) A new method for non-parametric multivariate analysis of variance.
\emph{Austral Ecology} \bold{26}, 32--46.

  Batagelj, V. (1988) Generalized Ward and related clustering problems. In H. Bock (Ed.),
  \emph{Classification and related methods of data analysis}, Amsterdam: North-Holland, pp. 67--74.
}
%\author{ ~~who you are~~ }
%\note{ ~~further notes~~ }

\seealso{
\code{\link{dissvar}} to compute the pseudo variance from dissimilarities and for a basic introduction to concepts of
pseudo variance analysis.
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

## R=1 implies no permutation test
da <- dissassoc(mvad.ham, group=mvad$gcse5eq, R=10)
print(da)
hist(da)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
