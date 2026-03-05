\name{rarcat}
\alias{rarcat}
\alias{plot.rarcat}
\alias{print.rarcat}
\alias{summary.rarcat}
\title{
Robustness Assessment of Regressions using Cluster Analysis Typologies (RARCAT)
}
\description{
\code{rarcat} is a wrapper for the functions \code{regressboot} and \code{bootpool} that performs the entire RARCAT procedure on all possible associations between a typology and covariates of interest. See Roth et al. (2024) or the R tutorial as \code{WeightedCluster} vignette for all details on the corresponding methods and their utility.
}
\usage{
rarcat(formula, data, diss, 
        robust=TRUE, R=500, 
        kmedoid=FALSE, hclust.method="ward.D", 
        fixed=FALSE, ncluster=10, cqi="HC",
        parallel=FALSE, progressbar=FALSE,
        fisher.transform=FALSE, 
		lmerCtrl=lme4::lmerControl())
\method{plot}{rarcat}(x, what="AME", covar=x$factorName[1], 
		pooled.ame=TRUE, naive.ame=TRUE,  
		with.legend=TRUE, legend.prop=NA, rows=NA, 
		cols=NA, main=NULL, 
		xlab=paste(covar, "Average Marginal Effect"),
		xlim=NULL, conf.level=0.95,...)
\method{print}{rarcat}(x, conf.level=0.95, single.row = FALSE, digits = 3, ...)
\method{summary}{rarcat}(object, ...)
}
\arguments{
  \item{formula}{A formula object with the clustering solution on the left side and the covariates of interest on the ride side.}
  \item{data}{The dataset (data frame) with column names corresponding to the information in formula. The number of individuals (row number) should match the dimension of \code{diss}.}
  \item{diss}{The numerical dissimilarity matrix used for clustering. Only a pre-computed matrix (i.e., where pairwise dissimilarities do not depend on the resample) is currently supported.}
  \item{robust}{Logical. TRUE (the default) indicates that RARCAT should be performed. FALSE implies a much faster function run but only output the original analysis, which is a standard regression analysis for all combinations of reference clusters and covariates.}
  \item{R}{The integer number of bootstrap. Set to 500 by default to attain a satisfactory precision around the estimates as the procedure involves multiple steps.}
  \item{kmedoid}{The clustering algorithm as a character string. Currently only "pam" (calling the function \code{wcKMedRange}) and "hierarchical" (calling the function \code{fastcluster::hclust}) are supported. By default "pam".}
  \item{hclust.method}{A character string with the method argument of \code{hclust}, "ward.D" by default.}
  \item{fixed}{Logical. TRUE implies that the number of clusters is the same in every bootstrap. FALSE (default) implies that an optimal number of clusters is evaluated each time.}
  \item{ncluster}{Integer. Either the number of clusters in every bootstrap if \code{fixed} is TRUE or the maximum number of clusters (starting from 2) to be evaluated in each bootstrap if \code{fixed} is FALSE.}
  \item{cqi}{A character string with the cluster quality index to be evaluated for each new partition. Any column of \code{as.clustrange} is supported, "CH" (the Calinski-Harabasz index) by default. Also works with \code{algo}= "pam".}
  \item{parallel}{Logical. Whether to initialize the parallel processing of the \code{future} package using the default \code{\link[future]{multisession}} strategy. If \code{FALSE} (default), then the current \code{\link[future]{plan}} is used. If \code{TRUE}, \code{\link[future]{multisession}} \code{\link[future]{plan}} is initialized using default values.}
  \item{progressbar}{Logical. Whether to initialize a progressbar using the \code{future} package. If \code{FALSE} (default), then the current progress bar \code{\link[progressr]{handlers}} is used . If \code{TRUE}, a new global progress bar \code{\link[progressr]{handlers}} is initialized.}
  \item{fisher.transform}{Logical. TRUE means that a Fisher transformation is applied in the multilevel model estimation step. This can be recommended in case of extreme associations (close to the -1 or 1 boundaries). FALSE by default.}
  \item{lmerCtrl}{Control parameter for lme4 (see \code{\link[lme4]{lmerControl}}}
  \item{x}{rarcat object to be printed or plotted.}
  \item{object}{rarcat object for summary (diagnostic tools).}
  \item{conf.level}{Confidence level for the confidence intervals. 0.95 by default.}
  \item{digits}{Number of significant digits to print (3 by default).}
  \item{single.row}{Logical. Whether to show confidence interval on the same or separate line (Default=FALSE).}
  \item{what}{Character. Information to plot. With "AME" (default), the boostrapped AME are shown. Set to "ranef" to view the distribution of observation-level random effect (usefull to identify potentially influential unstable observation). }
  \item{covar}{Character. The covariate of interest. }
  \item{pooled.ame}{Logical. Whether to add a vertical line and confidence interval for the pooled AME. }
  \item{naive.ame}{Logical. Whether to add a vertical line and confidence interval for the naive AME. }
  \item{with.legend}{Logical. If \code{FALSE}, the legend is not plotted.}
  \item{legend.prop}{Real in range [0,1]. Proportion of the graphic area devoted to the legend plot with.legend=TRUE. Default value is set according to the place (bottom or right of the graphic area) where the legend is plotted.}
  \item{rows}{Integers. Number of rows of the plot panel.}
  \item{cols}{Integers. Number of columns of the plot panel.}
  \item{main}{Character string. Title of the graphic. }
  \item{xlab}{x axis label.}
  \item{xlim}{Numerics. Limits of the x-axis.}
\item{\dots}{Additionnal parameters passed to/from methods.}
  
}
\details{
The \code{rarcat} function runs a standard typology-based association study and evaluates the impact of sampling uncertainty on the results, thus assessing the reproducibility of the analysis.
}
\value{
The output of \code{rarcattables} contains the following tables:
  
  The output of \code{bootpool} is a list with the following components:
  \item{nobs}{An integer with the number of observations (i.e., number of estimated AMES from the function \code{regressboot}) used to compute the robust estimates in the multilevel model. Due to missing observations when an individual does not appear in a bootstrap, \code{nobs < m} x \code{B}, where \code{m < M} is the number of individuals in a given cluster, \code{M} is the total number of individuals and \code{B} is the total number of bootstrap in \code{regressboot}.}
  \item{pooled.ame}{A numeric value indicating the pooled AME, which is the mean change in cluster membership probability for a change in the level of the covariate of interest over all bootstraps and all individuals belonging to the reference cluster in the original typology.}
  \item{standard.error}{Standard error of the pooled AME, which diminishes asymptotically as the number of bootstrap increases.}
  \item{bootstrap.stddev}{The estimate for the standard deviation of the bootstrap random effect. This can be used to construct a prediction interval for the association of interest (see Roth et al. 2024 for details on how to compute this).}
  \item{observation.stddev}{The estimate for the standard deviation of the bootstrap random effect.}
  \item{bootstrap.ranef}{A vector of size \code{B} containing the estimated random effects for each bootstrap.}
  \item{observation.ranef}{A vector of size \code{m} containing the estimated random effects for each observation in the reference cluster.}

  \item{original.analysis}{Average Marginal Effects (AMEs) estimated with multivariable logistic regressions and representing the expected change in the probability of belonging to a trajectory group (a reference cluster) for a change in the level of a variable (a covariate of interest), together with their confidence intervals.}
  \item{robust.analysis}{Pooled AMEs from the bootstrap procedure and their prediction intervals, representing the range of expected values if the clustering and associated regressions were performed on a new sample from the same underlying distribution. This table provide robust estimates for a typology-based association study.}
}
\references{
Roth, L., Studer, M., Zuercher, E., & Peytremann-Bridevaux, I. (2024). Robustness assessment of regressions using cluster analysis typologies: a bootstrap procedure with application in state sequence analysis. BMC medical research methodology, 24(1), 303. https://doi.org/10.1186/s12874-024-02435-8.
}
\author{
Leonard Roth
}
\examples{
## Loading the data (TraMineR package)
data(mvad)

## Reducing sample size to speed up computations
mvad <- mvad[1:200,]


## Creating the state sequence object
mvad.seq <- seqdef(mvad[, 17:86])

## Distance computation
diss <- seqdist(mvad.seq, method="LCS")

## Hierarchical clustering
hc <- fastcluster::hclust(as.dist(diss), method="ward.D")

## Computing cluster quality measures
clustqual <- as.clustrange(hc, diss=diss, ncluster=6)

## A six clusters solution is chosen here
mvad$clustering <- clustqual$clustering$cluster2

## The formula should include the typology (dependent) and the covariates of interest
## As in the original analysis, hierarchical clustering with Ward method is implemented
## The number of clusters is fixed to 2 here, larger values should often be used.
## For illustration purposes, the number of bootstrap is smaller than what it ought to be
rarcatout <- rarcat(clustering ~ Grammar + gcse5eq, mvad, diss, R = 30, 
                    kmedoid=TRUE, fixed = TRUE, ncluster = 2)

## Assess the robustness of the original analysis
rarcatout
#plot(rarcatout, covar="gcse5eqyes")
#plot(rarcatout, covar="gcse5eqyes", what="ranef")
#summary(rarcatout)
}
