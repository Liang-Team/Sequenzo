\name{seqnullcqi}
\alias{seqnullcqi}
\alias{plot.seqnullcqi}
\alias{print.seqnullcqi}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	 Sequence Analysis Typologies Validation Using Parametric Bootstrap 
}
\description{
\code{seqnullcqi} implements the methodology proposed by Studer (2021) for the validation of sequence analysis typologies using  parametric bootstraps. The method works by comparing the cluster quality of an observed typology with the quality obtained by clustering similar but nonclustered data. Several models to test the different structuring aspects of the sequences important in life-course research, namely, sequencing, timing, and duration (see function \code{\link{seqnull}}). This strategy allows identifying the key structural aspects captured by the observed typology. Plot and print methods of the \code{seqnullcqi} results are also provide. 
}
\usage{
seqnullcqi(seqdata, clustrange, R, model=c("combined", "duration", "sequencing", 
                    "stateindep", "Markov", "userpos"), seqdist.args=list(), 
					kmedoid = FALSE, hclust.method="ward.D", 
					parallel=FALSE, progressbar=FALSE, ...)
		   
\method{plot}{seqnullcqi}(x, stat, type = c("line", "density", "boxplot", "seqdplot"),
                          quant = 0.95, norm = TRUE, legendpos = "topright",
                          alpha = 0.2, ...)

\method{print}{seqnullcqi}(x, norm=TRUE, quant=0.95, digits=2, ...) 
} 
%- maybe also 'usage' for other objects documented here.
\arguments{
	\item{seqdata}{State sequence object of class \code{stslist}. The sequence data to use. Use \code{\link[TraMineR]{seqdef}} to create such an object.}
	\item{clustrange}{The clustering of the data to be validated as an object of class \code{clustrange}.	See \code{\link{as.clustrange}} or \code{\link{wcKMedRange}} to create such an object.}
	\item{model}{String. The model used to generate the similar but nonclustered data. It can be one of \code{"combined"}, \code{"duration"}, \code{"sequencing"}, \code{"stateindep"}, \code{"Markov"} or \code{"userpos"}. See \code{\link{seqnull}} for more information.}
	\item{R}{The number of bootstraps.}
	\item{seqdist.args}{List of arguments passed to \code{\link[TraMineR]{seqdist}} for computing the distances.}
	\item{kmedoid}{Logical. If \code{TRUE}, the PAM algorithm is used to cluster the data using  \code{\link{wcKMedRange}}. If \code{FALSE}, \code{hclust} is used.}
	\item{hclust.method}{String. Hierarchical method to use with \code{\link{hclust}}.}
	\item{x}{A \code{\link{seqnullcqi}} object to be plotted or printed.}
	\item{stat}{Character. The statistic to plot or "all" for all statistics. See \code{\link{wcClusterQuality}} for a list of possible values.}
	\item{type}{Character. The type of graphic to be plotted. If \code{type="line"} (default), a transparent line representing the cluster quality index for each bootstrap is plotted using a separate line. If \code{type="density"}, the density of the maximum cluster quality index values among the different number of groups is plotted as well as the original cluster quality values. If \code{type="beanplot"}, beanplot of the distribution of the cluster quality index values for each number of groups is plotted separately. If \code{type="seqdplot"}, a state distribution sequence plot of the sequences generated with the null model is plotted (see \code{\link[TraMineR]{seqdplot}}).}
	\item{quant}{Numeric. Quantile to use for the confidence intervals.}
	\item{norm}{Logical. If \code{TRUE}, cluster quality indices are standardized using the mean and standard deviation of the null distribution.}
	\item{legendpos}{Character. legend position, see \code{\link{legend}}.}
	\item{alpha}{Transparency parameter for the lines to be drawn (only for \code{type="line"}).}
	\item{digits}{Number of digits to be printed.}
	\item{parallel}{Logical. Whether to initialize the parallel processing of the \code{future} package using the default \code{\link[future]{multisession}} strategy. If \code{FALSE} (default), then the current \code{\link[future]{plan}} is used. If \code{TRUE}, \code{\link[future]{multisession}} \code{\link[future]{plan}} is initialized using default values.}
	\item{progressbar}{Logical. Whether to initialize a progressbar using the \code{future} package. If \code{FALSE} (default), then the current progress bar \code{\link[progressr]{handlers}} is used . If \code{TRUE}, a new global progress bar \code{\link[progressr]{handlers}} is initialized.}
	\item{\dots}{Additionnal parameters passed to \code{\link{seqnull}} (for \code{seqnullcqi}) or \code{\link{plot}} or \code{\link{print}}.}
}
\details{
The \code{seqnullcqi} function provides a validation method for sequence analysis typologies using parametric bootstraps as proposed in Studer (2021). This method works by comparing the value of the cluster quality of an observed typology with the cluster quality obtained by clustering similar but nonclustered data. More precisely it works as follows. 

\enumerate{
	\item Cluster the observed sequence data and compute the associated cluster quality indices. 
	\item Repeat \code{R} times:
	\enumerate{
		\item Generate similar but nonclustered data using a \emph{null} model (see \code{\link{seqnull}} for available \emph{null} models).
		\item Cluster the generated data using the same distance measure and clustering algorithm as in step 1. 
		\item Record the quality indices values of this null clustering. 
	}
	\item Compare the quality of the observed typology with the one obtained in the \code{R} bootstraps with the \emph{null} sequence data using plot and print methods. 
	\item If the cluster quality measure of the observed typology is constantly higher than the ones obtained with \emph{null} data, a \dQuote{good} typology has been found.
}

Several \emph{null} models are provided to test the different structuring aspects of the sequences important in life-course research, namely, sequencing, timing, and duration (see function \code{\link{seqnull}} and Studer, 2021). This strategy allows identifying the key structural aspects captured by the observed typology.
}
\value{
\code{seqnullcqi} returns a \code{"seqnullcqi"} object with the following components:
\item{seqdata}{The sequence data generated by the null model (see \code{\link{seqnull}}}
\item{stats}{The cluster quality indices for the null data.}
\item{clustrange}{The clustering of the data to be validated as an object of class \code{clustrange}.}
\item{R}{The number of bootstraps}
\item{kmedoid}{Logical. If \code{TRUE}, the PAM algorithm was used to cluster the data using  \code{\link{wcKMedRange}}.}
\item{hclust.method}{Hierarchical method to used with \code{\link{hclust}}.}
\item{seqdist.args}{List of arguments passed to \code{\link[TraMineR]{seqdist}} for computing the distances.}
\item{nullmodel}{List of arguments passed to \code{\link{seqnull}} to generate the sequence data under the null model.}
}

\references{
Studer, M. (2021). Validating Sequence Analysis Typologies Using Parametric Bootstrap. \emph{Sociological Methodology}. \doi{10.1177/00811750211014232}

A brief introduction to the \code{R} code needed to use parametric bootstraps for typology validation in sequence analysis is provided here \url{https://sequenceanalysis.org/2023/10/19/validating-sequence-analysis-typologies-using-parametric-bootstrap/}
}

\seealso{
	See Also \code{\link{seqnull}} for description of the null models.
}
\examples{
data(biofam)

## Create the sequence object
bf.seq <- seqdef(biofam[sample.int(nrow(biofam), 100),10:25])

## Library fastcluster greatly improve computation time when using hclust
# library(fastcluster)
## Computing distances
diss <- seqdist(bf.seq, method="HAM")
## Hierarchical clustering
hc <- hclust(as.dist(diss), method="ward.D")
# Computing cluster quality measures.
clustqual <- as.clustrange(hc, diss=diss, ncluster=7)

# Compute cluster quality measure for the null model "combined"
# seqdist.args should be the same as for seqdist above except the sequence data.
# Clustering methods should be the same as above.
bcq <- seqnullcqi(bf.seq, clustqual, R=5, model=c("combined"), 
				seqdist.args=list(method="HAM"),
				hclust.method="ward.D")

# Print the results
bcq

## Different kind of plots

plot(bcq, stat="ASW", type="line")
plot(bcq, stat="ASW", type="density")
plot(bcq, stat="ASW", type="boxplot")

}
