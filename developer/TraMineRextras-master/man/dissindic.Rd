\name{dissindic}
\alias{dissindic}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Sequence marginality and gain indicators.
%Compute the marginality and gain indicators of a relationship between a covariate and a distance matrix.
}
\description{
The marginality and gain indicators measure the contribution of each observation to a quantitative relationship between a covariate and the sequences using a distance matrix.  These indicators allow situating cases according to this quantitative relationship. The marginality indicator quantifies the typicality of each case within each group of the explanatory covariate using a measure of distance between cases and gravity centers. The gain indicator aims to identify cases that are either illustrative of, or discordant with, the quantitative association. It is computed as the contribution of each case to the explained sum of squares, i.e. the sum of squares explained by the covariate.
}
\usage{
dissindic(diss, group, gower = FALSE, squared = FALSE, weights = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{diss}{A dissimilarity matrix or a \code{dist} object.}
  \item{group}{A categorical variable.}
  \item{gower}{Logical: Is the dissimilarity matrix already a Gower matrix?}
  \item{squared}{Logical: Should we square the provided dissimilarities?}
  \item{weights}{Optional numerical vector of case weights.}
}
\details{

The two indicators are computed within the discrepancy analysis framework (see \code{\link[TraMineR]{dissmfacw}}). The marginality is computed as the "residual" of the discrepancy analysis. A high value means that a sequence (or another object) is far from the center of gravity of its group, i.e. the most typical situation. A low value indicates a sequence (or another object) close to this gravity center. 

By combining the "residuals" of the null model (without covariate) and the marginality, we can identify sequences that are better represented when using the covariate than without it. These values measure the contributions of a sequence to the between or explained sums of squares, a concept directly linked to the explained discrepancy. The gain therefore measures the statistical gain of information for each case when taking the covariate into account.

The so-called Gower matrix is a transformation of the distance matrix required to compute the explained, residual and total sum of squares. The distance matrix (argument \code{diss}) is automatically transformed to a Gower matrix unless the argument \code{gower=TRUE}. This option can be used to avoid multiple transformation of the distance matrix to the Gower matrix which can be time-consuming, but it requires the user to provide a Gower matrix using the \code{diss} argument. The function \code{gower_matrix} in the TraMineR package can be used to compute it from a distance matrix.
}
\value{A \code{data.frame} containing three columns: 

	\item{group}{The categorical variable used.}
	\item{marginality}{The value of the marginality indicator.}
	\item{gain}{The value of the gain indicator .}

}
\references{
Le Roux, G., M. Studer, A. Bringé, C. Bonvalet (2023). Selecting Qualitative Cases Using Sequence Analysis: A Mixed-Method for In-Depth Understanding of Life Course Trajectories, \emph{Advances in Life Course Research}, \doi{10.1016/j.alcr.2023.100530}.

Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

}
\author{Matthias Studer}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link[TraMineR]{dissvar}} to compute a pseudo variance from dissimilarities and for a basic introduction to concepts of discrepancy analysis.
   \cr
   \code{\link[TraMineR]{dissassoc}} to test association between objects represented by their dissimilarities and a covariate.
\cr
\code{\link[TraMineR]{dissmfacw}} to perform multi-factor analysis of variance from pairwise dissimilarities.

}
\examples{
## Defining a state sequence object
data(mvad)
mvad <- mvad[1:100, ] ## Use a subsample to avoid long computation time
mvad.seq <- seqdef(mvad[, 17:86])

## Building dissimilarities (any dissimilarity measure can be used)
mvad.ham <- seqdist(mvad.seq, method="HAM")

## Study association with 
di <- dissindic(mvad.ham, group=mvad$gcse5eq)

## Plot sequences sorted by gain, illustrative trajectories at the top 
## and counterexample at the bottom
seqIplot(mvad.seq, group=mvad$gcse5eq, sortv=di$gain)

## Plot sequences sorted by marginality, central trajectories at the bottom
seqIplot(mvad.seq, group=mvad$gcse5eq, sortv=di$marginality)

##Scatterplot of the indicators separated by group value 
## as in Le Roux, et al. 2023
par(mfrow=c(1, 2))
## Plot for the "no" category
plot(di$gain[mvad$gcse5eq=="no"], di$marginality[mvad$gcse5eq=="no"], main="No gcseq5q", 
	xlim=range(di$gain), ylim=range(di$marginality))
abline(h=mean(di$marginality), v=0) ## Draw reference lines
plot(di$gain[mvad$gcse5eq=="yes"], di$marginality[mvad$gcse5eq=="yes"], main="Yes gcseq5q", 
	xlim=range(di$gain), ylim=range(di$marginality))
abline(h=mean(di$marginality), v=0) ## Draw reference lines

}

