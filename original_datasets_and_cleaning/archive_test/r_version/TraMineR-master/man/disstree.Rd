\name{disstree}
\alias{disstree}
\alias{print.disstree}
\title{Dissimilarity Tree}
\description{
  Tree structured discrepancy analysis of objects described by their pairwise dissimilarities.
}
\usage{
disstree(formula, data = NULL, weights = NULL, min.size = 0.05,
  max.depth = 5, R = 1000, pval = 0.01, object = NULL,
  weight.permutation = "replicate", squared = FALSE, first = NULL,
  minSize, maxdepth)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{Formula with a dissimilarity matrix as left hand side and the candidate partitioning variables on the right side.}
  \item{data}{Data frame where variables in formula will be searched for.}
  \item{weights}{Optional numerical vector of weights.}
  \item{min.size}{Minimum number of cases in a node, will be treated as a proportion if less than 1.}
  \item{max.depth}{Maximum depth of the tree}
  \item{R}{Number of permutations used to assess the significance of the split.}
  \item{pval}{Maximum allowed p-value for a split}
  \item{object}{An optional R object represented by the dissimilarity matrix. This object may be used by the \code{print} method or  \code{\link{disstree2dot}} to render specific object type.}
  \item{weight.permutation}{Weight permutation method: \code{"diss"} (attach weights to the dissimilarity matrix), \code{"replicate"} (replicate cases using \code{weights}), \code{"rounded-replicate"} (replicate case using rounded \code{weights}), \code{"random-sampling"} (random assignment of covariate profiles to the objects using distributions defined by the weights.)}
  \item{squared}{Logical: Should the \code{diss} dissimilarities be squared?}
  \item{first}{One of the variable in the right-hand side of the formula. This forces the first node of the tree to be split by this variable.}
  \item{minSize}{Deprecated. Use \code{min.size} instead.}
  \item{maxdepth}{Deprecated. Use \code{max.depth} instead.}
}

\details{
  The procedure iteratively splits the data. At each step, the procedure selects the variable and split that explain the greatest part of the discrepancy,
  i.e., the split for which we get the highest pseudo R2.
  The significance of the retained split is assessed through a permutation test.

  \code{\link{seqtree}} provides a simpler interface if you plan to use \code{disstree} for state sequence objects.
}
\value{
An object of class \code{disstree} that contains the following components:
  \item{root}{A node object, root of the tree}
  \item{info}{General information such as parameters used to build the tree}
  \item{info$adjustment}{A \code{\link{dissassoc}} object providing global statistics for tree.}
  \item{formula}{The formula used to generate the tree}
  \item{data}{data used to build the tree}
  \item{weights}{weights}
}
\references{
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010)
  Discrepancy	analysis of complex objects using dissimilarities.
  In F. Guillet, G. Ritschard, D. A. Zighed and H. Briand (Eds.),
  \emph{Advances in Knowledge Discovery and Management},
  Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.

  Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2009)
  Analyse de dissimilarités par arbre d'induction. In EGC 2009,
  \emph{Revue des Nouvelles Technologies de l'Information}, Vol. E-15, pp. 7-18.

  Anderson, M. J. (2001) A new method for non-parametric multivariate analysis of variance.
\emph{Austral Ecology} \bold{26}, 32-46.

  Batagelj, V. (1988) Generalized ward and related clustering problems. In H. Bock (Ed.),
  \emph{Classification and related methods of data analysis}, Amsterdam: North-Holland, pp. 67-74.

  Piccarreta, R. et F. C. Billari (2007) Clustering work and family trajectories by using a divisive
algorithm. \emph{Journal of the Royal Statistical Society A} \bold{170}(4), 1061--1078.
}
%\author{}
\seealso{
	\code{\link{seqtree}} to generate a specific \code{disstree} objects for analyzing state sequences.
   \cr
	\code{\link{seqtreedisplay}} to generate graphic representation of \code{seqtree} objects when analyzing state sequences.
   \cr
   \code{\link{disstreedisplay}} is a more general interface to generate such representation for other type of objects.
   \cr
   \code{\link{disstreeleaf}} to get leaf membership of each case.
   \cr
   \code{\link{disstree.get.rules}} to get the list of classification rules as R commands.
   \cr
   \code{\link{disstree.assign}} for the index of the rules applying to provided profiles.
   \cr
   \code{\link{dissvar}} to compute discrepancy using dissimilarities and for a basic introduction to discrepancy analysis.
   \cr
   \code{\link{dissassoc}} to test association between objects represented by their dissimilarities and a covariate.
   \cr
    \code{\link{dissmfacw}} to perform multi-factor analysis of variance from pairwise dissimilarities.
\cr
\code{\link{disscenter}} to compute the distance of each object to its group center from pairwise dissimilarities.

}
\examples{
data(mvad)

## Defining a state sequence object
mvad.seq <- seqdef(mvad[, 17:86])

## Computing dissimilarities (any dissimilarity measure can be used)
mvad.ham <- seqdist(mvad.seq, method="HAM")
## Grow the tree using a low R value for illustration.
## For R=10, pval cannot be lower than 0.1
dt <- disstree(mvad.ham~ male + Grammar + funemp + gcse5eq + fmpr + livboth,
               data=mvad, R = 10, pval = 0.1)
print(dt)


## Will only work if GraphViz is properly installed
## See seqtree for a simpler way to plot a sequence tree.
\dontrun{
disstreedisplay(dt, image.fun = seqdplot, image.data = mvad.seq,
                ## Additional parameters passed to seqdplot
                with.legend = FALSE, xaxis = FALSE, ylab = "", border=NA)
}

## Second method, using a specific function
myplotfunction <- function(individuals, seqs, ...) {
  par(font.sub=2, mar=c(3,0,6,0), mgp=c(0,0,0))
  ## using mds to order sequence in seqIplot
  mds <- suppressMessages(cmdscale(seqdist(seqs[individuals,], method="HAM"),k=1))
  seqIplot(seqs[individuals,], sortv=mds,...)
}

## If image.data is not set, indexes of individuals are sent to image.fun
\dontrun{
disstreedisplay(dt, image.fun = myplotfunction, cex.main = 3,
                ## additional parameters passed to myplotfunction
                seqs = mvad.seq,
                ## additional parameters passed to seqIplot (through myplotfunction)
                with.legend = FALSE, xaxis = FALSE, ylab = "")
}

## Retrieving terminal node membership
term.leaf <- disstreeleaf(dt)
table(term.leaf)

## Retrieving classification rules
rules <- disstree.get.rules(dt)

## Index of rule (terminal leaf) that applies to a specified profile
## covariates are: male, Grammar, funemp, gcse5eq, fmpr, livboth

profile <- data.frame(male="no", Grammar="yes", funemp="no", gcse5eq="yes", fmpr="no", livboth="no")
rules[disstree.assign(rules, profile=profile)]

}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
