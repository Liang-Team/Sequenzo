\name{clustassoc}
\alias{clustassoc}
\alias{plot.clustassoc}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
 Share of an association between an object (described by a dissimilarity matrix) and a covariate that is reproduced by a clustering solution.
}
\description{

 The \code{clustassoc} measures to which extent a clustering solution can account for the relationship between a covariate and the objects of interest, i.e. the sequences or any other object described by a dissimilarity matrix. It can be used to guide the choice of the number of groups ensuring that the clustering captures the relevant information to account for a statistical relationship of interest. This is useful when the clustering is used in subsequent analyses, such as regressions. In this case, the within-cluster variation is ignored, as objects clustered together are described by a single value. Ensuring that the association is accounted for by the clustering can avoid drawing wrong conclusions (see Unterlerchner et al. 2023). 
}
\usage{
clustassoc(clustrange, diss, covar, weights = NULL)
\method{plot}{clustassoc}(x, stat=c("Unaccounted", "Remaining", "BIC"), type="b", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{clustrange}{A \code{clustrange} object regrouping the different clustering solutions to be evaluated. }
  \item{diss}{A dissimilarity matrix or a dist object (see \code{\link{dist}}).}
  \item{covar}{Vector (Numeric or factor): the covariate of interest. The type of the vector matters for the computation of the BIC (see details). If Numeric, a linear regression is used, while a multinomial regression is used for categorical/factor variables.}
  \item{weights}{Optional numerical vector containing weights.}
  \item{x}{A \code{clustassoc} object to be plotted.}
  \item{stat}{The information to be plotted according to the number of groups. \code{"Unaccounted"} (default) plots the share of the association that is NOT accounted for by the clustering solution. \code{"Remaining"} plots the share of the overall variability/discrepancy of the object remaining when controlling for the clustering. \code{"BIC"} plots the BIC of a regression predicting the covariate using the clustering solution (see details).}
  \item{type}{character indicating the type of plotting (see \code{\link{plot.default}}). \code{"b"} plots points and lines.}
  \item{\dots}{Additionnal parameters passed to/from methods.}
}
\details{
The \code{clustassoc} measures to which extent a clustering solution can account for the relationship between a covariate and the objects of interest. It can be used to guide the choice of the number of groups of the clustering to ensure that it captures the relevant information to account for a statistical relationship of interest. 

The method works as follows. The relationship between trajectories (or any objects described by a distance matrix) and covariates can be studied directly using discrepancy analysis (see Studer et al. 2011). It measures the strength of the relationship with a Pseudo-R2, measuring the share of the variation of the object explained by a covariate. The method works without prior clustering, and therefore, without data simplification. The method is provided by the \code{\link[TraMineR]{dissmfacw}} function from the \code{TraMineR} package.

Multifactor discrepancy analysis allows measuring a relationship while controlling for other covariates. the \code{clustassoc} function measures the remaining association between the objects and the covariate while controlling for the clustering. If the covariate Pseudo-R2 remains high (or at the same level), it means that the clustering does not capture the relationship between covariates and the objects. In other words, the clustering has simplified the relevant information to capture this relationship. Conversely, if the Pseudo-R2 is much lower, it means that the clustering reproduces the key information to understand the relationship. Using this strategy, the \code{clustassoc} measure the share of the original Pseudo-R2 that is taken into account by our clustering.

The function also compute the BIC of a regression predicting the covariate using the clustering solution as proposed by Han et al. 2017. A lower BIC is to be preferred. The method is, however, less reliable than the previous one.
}
\value{
A \code{clustassoc} object containing the following information for each clustering:
    \item{Unaccounted}{The share of the original association that is NOT accounted for by the clustering solution.}
    \item{Remaining}{The remaining strength of the association (share of the variability of the object) that is not accounted for by the clustering solution.}
    \item{BIC}{The BIC of a model explaining the covariate using the clustering as explanatory variable. }
    \item{Remaining}{The remaining strength of the association (share of the variability of the object) that is not accounted for by the clustering solution.}
    \item{numcluster}{The number of clusters (and 1 means no clustering).}
%% ...
}
\references{

Unterlerchner, L., M. Studer and A. Gomensoro (2023). Back to the Features. Investigating the Relationship Between Educational Pathways and Income Using Sequence Analysis and Feature Extraction and Selection Approach. \emph{Swiss Journal of Sociology}.

Studer, M. 2013. WeightedCluster Library Manual: A Practical Guide to Creating Typologies of Trajectories in the Social Sciences with R.\emph{LIVES Working Papers 2013(24)}: 1-32. 

Studer, M., G. Ritschard, A. Gabadinho and N. S. Mueller (2011). Discrepancy analysis of state sequences, \emph{Sociological Methods and Research}, Vol. 40(3), 471-510, \doi{10.1177/0049124111415372}.

Han, Y., A. C. Liefbroer and C. H. Elzinga. 2017. Comparing Methods of Classifying Life Courses: Sequence Analysis and Latent Class Analysis. \emph{Longitudinal and Life Course Studies 8(4)}: 319-41. 

}
\author{
Matthias Studer
}

\seealso{
See Also \code{\link{as.clustrange}} for cluster quality indexes, and the \code{\link[TraMineR]{dissmfacw}} function from the \code{TraMineR} package.
}
\examples{
data(mvad)

## Small subsample to reduce computations
mvad <- mvad[1:50,]

## Sequence object
mvad.seq <- seqdef(mvad[, 17:86])

## Compute distance using Hamming distance
diss <- seqdist(mvad.seq, method="HAM")

## Ward clustering
wardCluster <- hclust(as.dist(diss), method="ward.D")

## Computing clustrange from Ward clustering up to 5 groups
wardRange <- as.clustrange(wardCluster, diss=diss, ncluster=5)

## Compute clustassoc
## How many groups are required to account for the relationship 
## between trajectories and the gcse5eq covariate 
assoc <- clustassoc(wardRange, covar=mvad$gcse5eq, diss=diss)

## Plot unaccounted share of the association 
## A value close to zero means that the relationship is accounted for.
## Here at least 2-4 groups are required
plot(assoc)

## Plot BIC
## A low value means that an association between trajectories and the covariate is identified.
## 2-3 groups show best results.
plot(assoc, stat="BIC")


## Plot remaining share of the variability of the sequences not explained by clustering
## A value close to zero means that there is no association left (similar)
## Here at least 2-4 groups are required
plot(assoc, stat="Remaining")

}
