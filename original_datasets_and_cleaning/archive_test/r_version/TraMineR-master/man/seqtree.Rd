\name{seqtree}
\alias{seqtree}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Tree structured analysis of a state sequence object.
}
\description{
Facility for growing a regression tree for a state sequence object.
}
\usage{
seqtree(formula, data = NULL, weighted = TRUE, min.size = 0.05,
  max.depth = 5, R = 1000, pval = 0.01, weight.permutation = "replicate",
  seqdist.args = list(method = "LCS", norm = "auto"), diss = NULL,
  squared = FALSE, first = NULL, minSize, maxdepth, seqdist_arg)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a formula where the left hand side is a state sequence object (see \code{\link{seqdef}})  and the right hand specifies the candidate variables for partitioning the set of sequences.}
  \item{weighted}{Logical. If \code{TRUE}, use the weights of the state sequence object.}
  \item{data}{a data frame where variables in the formula will be searched}
  \item{min.size}{minimum number of cases in a node, in percentage if less than 1.}
  \item{max.depth}{maximum depth of the tree.}
  \item{R}{Number of permutations used to assess the significance of the split.}
  \item{pval}{Maximum p-value, in percent.}
  \item{weight.permutation}{Weights permutation method: "diss" (attach weights to the dissimilarity matrix), "replicate" (replicate case according to the \code{weights} arguments), "rounded-replicate" (replicate case according to the rounded \code{weights} arguments), "random-sampling" (random assignment of covariate profiles to the objects using distributions defined by the weights.)}
  \item{seqdist.args}{list of arguments directly passed to \code{\link{seqdist}}, only used if \code{diss=NULL}}
  \item{diss}{An optional dissimilarity matrix. If not provided, a dissimilarity matrix is computed using \code{\link{seqdist}} and \code{seqdist.args}}
  \item{squared}{Logical. If \code{TRUE}, the dissimilarity matrix is squared}
  \item{first}{Character. An optional variable name to force the first split.}
  \item{minSize}{Deprecated. Use \code{min.size} instead.}
  \item{maxdepth}{Deprecated. Use \code{max.depth} instead.}
  \item{seqdist_arg}{Deprecated. Use \code{seqdist.args} instead.}
}
\details{
The function provides a simplified interface for applying \code{\link{disstree}} on state sequence objects.

The \code{seqtree} objects can be "plotted" with \code{\link{seqtreedisplay}}. A print method is also available which prints the medoid sequence for each terminal node.
}
\value{
    A \code{seqtree} object with same attributes as \code{\link{disstree}} objects.

    The leaf membership is in the first column of the fitted attribute. For example, the leaf memberships for a tree \code{dt}  are in \code{dt$fitted[,1]}.
}

\references{
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.
}

\seealso{
    \code{\link{seqtreedisplay}}, \code{\link{disstree}}
}


\examples{
data(mvad)

## Defining a state sequence object
mvad.seq <- seqdef(mvad[, 17:86])

## Growing a seqtree from Hamming distances:
##   Warning: The R=10 used here to save computation time is
##   much too small and will generate strongly unstable results.
##   We recommend to set R at least as R=1000.
##   To comply with this small R value, we set pval = 0.1.
seqt <- seqtree(mvad.seq~ male + Grammar + funemp + gcse5eq + fmpr + livboth,
    data=mvad, R=10, pval=0.1, seqdist.arg=list(method="HAM", norm="auto"))
print(seqt)

## Growing a seqtree from an existing distance matrix
mvad.dhd <- seqdist(mvad.seq, method="DHD")
seqt <- seqtree(mvad.seq~ male + Grammar + funemp + gcse5eq + fmpr + livboth,
    data=mvad, R=10, pval=0.1, diss=mvad.dhd)
print(seqt)


### Following commands only work if GraphViz is properly installed
\dontrun{
seqtreedisplay(seqt, type="d", border=NA)
seqtreedisplay(seqt, type="I", sortv=cmdscale(mvad.dhd, k=1))
}

}

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
