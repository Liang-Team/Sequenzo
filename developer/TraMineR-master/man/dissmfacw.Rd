\name{dissmfacw}
\alias{dissmfacw}
\alias{gower_matrix}
\alias{print.dissmultifactor}
\title{Multi-factor ANOVA from a dissimilarity matrix}
\description{
  Perform a multi-factor analysis of variance from a dissimilarity matrix.
}
\usage{
dissmfacw(formula, data, R = 1000, gower = FALSE, squared = FALSE,
    weights = NULL)
    
gower_matrix(diss, squared=TRUE, weights=NULL)

\method{print}{dissmultifactor}(x, pvalue.confint=0.95, digits = NULL, ...)
}

\arguments{
  \item{formula}{A regression-like formula. The left hand side term should be a dissimilarity matrix or a \code{dist} object.}
  \item{data}{A data frame from which the variables in \code{formula} should be taken.}
  \item{R}{Number of permutations used to assess significance.}
  \item{gower}{Logical: Is the dissimilarity matrix already a Gower matrix?}
  \item{squared}{Logical: Should we square the provided dissimilarities?}
  \item{weights}{Optional numerical vector of case weights.}
  \item{diss}{Dissimilarity matrix}
  \item{x}{a \code{dissmultifactor} object as returned by \code{dissmfacw}}
  \item{pvalue.confint}{Real in range [0,1]. Confidence probability. }
  \item{digits}{Integer or \code{NULL}. Number of digits.}
  \item{...}{Other generic print arguments.}
}
\details{
  Function \code{dissmfacw} is, in some way, a generalization of \code{\link{dissassoc}} to account for several explanatory variables.
  The function computes the part of discrepancy explained by the list of covariates specified in the \code{formula}.
  It provides for each covariate the Type-II effect, i.e. the effect measured when removing the covariate from the full model with all variables included.

  (The returned F values may slightly differ from those obtained with TraMineR versions older than 1.8-9. Since 1.8-9, the within sum of squares at the denominator is divided by \eqn{n-m} instead of \eqn{n-m-1}, where \eqn{n} is the sample size and \eqn{m} the total number of predictors and/or contrasts used to represent categorical factors.)

  For a single factor \code{dissmfacw} is slower than \code{\link{dissassoc}}.
  Moreover, the latter performs also tests for homogeneity in within-group discrepancies (equality of variances) with a generalization of Levene's and Bartlett's statistics.

  Part of the function is based on the Multivariate Matrix Regression with qr decomposition algorithm written in SciPy-Python by Ondrej Libiger and Matt Zapala (See \cite{Zapala and Schork}, 2006, for a full reference.)  The algorithm has been adapted for Type-II effects and extended to account for case weights.
  
  Function \code{gower_matrix} transforms the provided dissimilarity matrix into a Gower matrix.
}

\value{
  A \code{dissmultifactor} object with the following components:
  \item{mfac}{The part of variance explained by each variable (comparing full model to model without the specified variable) and its significance using permutation test}
  \item{call}{Function call}
  \item{perms}{Permutation values as a \code{boot} object}
}
\references{
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010)
  Discrepancy analysis of complex objects using dissimilarities.
  In F. Guillet, G. Ritschard, D. A. Zighed and H. Briand (Eds.),
  \emph{Advances in Knowledge Discovery and Management},
  Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2009).
  Analyse de dissimilarités par arbre d'induction. In EGC 2009,
  \emph{Revue des Nouvelles Technologies de l'Information}, Vol. E-15, pp. 7-18.

  Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance.
\emph{Austral Ecology} 26, 32-46.

  McArdle, B. H. and M. J. Anderson (2001). Fitting multivariate models to community data: A
comment on distance-based redundancy analysis. \emph{Ecology} 82(1), 290-297.

  Zapala, M. A. and N. J. Schork (2006). Multivariate regression analysis of distance matrices for
  testing associations between gene expression patterns and related variables. \emph{Proceedings of
the National Academy of Sciences of the United States of America} 103(51), 19430-19435.
}

\seealso{
\code{\link{dissvar}} to compute a pseudo variance from dissimilarities and for a basic introduction to concepts of discrepancy analysis.
   \cr
   \code{\link{dissassoc}} to test association between objects represented by their dissimilarities and a covariate.
\cr
\code{\link{disstree}} for an induction tree analysis of objects characterized by a dissimilarity matrix.
\cr
\code{\link{disscenter}} to compute the distance of each object to its group center from pairwise dissimilarities.
}
\examples{
## Define the state sequence object
data(mvad)
mvad.seq <- seqdef(mvad[, 17:86])
## Here, we use only first 100 sequences
mvad.seq <- mvad.seq[1:100,]

## Compute dissimilarities (any dissimilarity measure can be used)
mvad.ham <- seqdist(mvad.seq, method="HAM")

## And now the multi-factor analysis
print(dissmfacw(mvad.ham ~ male + Grammar + funemp +
	gcse5eq + fmpr + livboth, data=mvad[1:100,], R=10))
}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
