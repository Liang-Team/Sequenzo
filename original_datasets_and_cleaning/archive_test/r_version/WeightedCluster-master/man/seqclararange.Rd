\name{seqclararange}
\alias{seqclararange}


\title{
	CLARA Clustering for Sequence Analysis
}
\description{

Cluster large databases of sequences for a different number of groups using the CLARA algorithm based on subsampling to reduce computational burden. Crisp, fuzzy and representativeness clustering are available. The function further computes several cluster quality measures. 

}
\usage{
seqclararange(seqdata, R = 100, sample.size = 40 + 2 * max(kvals), 
  kvals = 2:10,  seqdist.args = list(method = "LCS"),
  method=c("crisp", "fuzzy", "representativeness", "noise"), 
  m = 1.5, criteria = c("distance"), stability = FALSE, dnoise=NULL,
  parallel = FALSE, progressbar = FALSE, keep.diss = FALSE, 
  max.dist = NULL)
}

\arguments{
  \item{seqdata}{State sequence object of class \code{stslist}. The sequence data to use. Use \code{\link[TraMineR]{seqdef}} to create such an object.}
  \item{R}{Numeric. The number of subsamples to use.}
  \item{sample.size}{Numeric. The size of the subsamples, the default values is the one proposed by Kaufmann and Rousseuuw (1990). However, larger values (typically between 1000 and 10 000) are recommended.}
  \item{kvals}{Numeric vector. The different number of groups to compute. }
  \item{seqdist.args}{List of arguments passed to \code{\link[TraMineR]{seqdist}} for computing the distances.} 
  \item{method}{Character. The clustering approach to use, with default to "crisp" clustering. "fuzzy", "noise" or "representativeness" approaches can also be used.}
  \item{m}{Numeric. Only used for fuzzy clustering, the value of the fuzzifier. }
  \item{criteria}{Character. The name of the criteria used for selecting the best clustering among the different runs. The following values are accepted: "distance" (Default, average value to cluster medoids),  "db" (Davies-Bouldin Index), "xb" (Xie-Beni index), "pbm" (PBM Index), "ams" (Average medoid silhouette value). }
  \item{stability}{Logical. If \code{TRUE}, stability measures are computed (can be time consuming, especially for fuzzy clustering). Default to \code{FALSE}.}
  \item{dnoise}{Numerical. The theoretically defined distance to the noise cluster. Mandatory for noise clustering.}
  \item{parallel}{Logical. Whether to initialize the parallel processing of the \code{future} package using the default \code{\link[future]{multisession}} strategy. If \code{FALSE} (default), then the current \code{\link[future]{plan}} is used. If \code{TRUE}, \code{\link[future]{multisession}} \code{\link[future]{plan}} is initialized using default values.}
	\item{progressbar}{Logical. Whether to initialize a progressbar using the \code{future} package. If \code{FALSE} (default), then the current progress bar \code{\link[progressr]{handlers}} is used . If \code{TRUE}, a new global progress bar \code{\link[progressr]{handlers}} is initialized.}
  \item{keep.diss}{Logical. Whether to keep the distances to the medoids. Set to \code{FALSE} by default.}
  \item{max.dist}{Numeric. Maximal theoretical distance value between sequences. Required for \code{method="representativeness"} clustering.}

}
\details{
\code{seqclararange} relies on the CLARA algorithm to cluster large database. The algorithm works as follows. 

\enumerate{
  \item Randomly take a subsample of the data of size \code{sample.size}.
  \item Cluster the subsample using the PAM algorithm initialized using Ward to speed up the computations (see \code{wcKmedoids}).
  \item Use the identified medoids to assign cluster membership in the whole dataset.
  \item Evaluate the resulting clustering using a \code{criteria} (see argument), the average distances to the medoids by default.
}

These steps are repeated \code{R} times and the best solution according to the given criterion is kept.


To minimize the computation, the operation is repeated for different number of groups, which then allows to choose the best number of groups according to different cluster quality indices. The following indices are computed automatically: \code{"Avg dist"} (Average distance to cluster medoids), \code{"PBM"}(PBM Index), \code{"DB"} (Davies-Bouldin Index), \code{"XB"} (Xie-Beni Index), \code{"AMS"} (Average medoid silhouette width), \code{"ARI>0.8"} (Number of iteration similar to the current best, only if \code{stability=TRUE}, \code{"JC>0.8"} (Number of iteration similar to the current best, only if \code{stability=TRUE}.

}
\value{
	A \code{seqclararange} object with the following components:
	\item{kvals:}{The different number of groups evaluated.}
	\item{clustering:}{The retained clustering for each number of groups. For \code{"crisp"} clustering, a \code{data.frame} with the clustering in column named clusterX, with X the number of groups. For \code{"fuzzy"} and \code{"representativeness"}, a list of membership matrix, with each elements named clusterX, with X the number of groups.}
	\item{\code{stats}:}{A \code{matrix} containing the clustering statistics of each cluster solution.}
	\item{clara:}{Detailed information on the best clustering for each number of groups, in the same order as kvals.}
}
\references{
	Studer, M., R. Sadeghi and L. Tochon (2024). Sequence Analysis for Large Databases. \emph{LIVES Working Papers 104} \doi{10.12682/lives.2296-1658.2024.104}

}



\seealso{
See also as \code{\link{plot.seqclararange}} to plot the results. 
}
\examples{
data(biofam) #load illustrative data
## Defining the new state labels 
statelab <- c("Parent", "Left", "Married", "Left/Married",  "Child", 
            "Left/Child", "Left/Married/Child", "Divorced")
## Creating the state sequence object,
biofam.seq <- seqdef(biofam[1:100, 10:25], alphabet=0:7, states=statelab)



## Clara clustering
bfclara <- seqclararange(biofam.seq, R = 3, sample.size = 10, kvals = 2:3, 
  seqdist.args = list(method = "HAM"), parallel=FALSE, 
  stability=TRUE)


#Show the cluster quality measures.
bfclara
#Plot and normalize the values for easier identification of minimum and maximum values.
plot(bfclara, norm="range")
## Stability values.
plot(bfclara, stat="stabmean")
plot(bfclara, stat="stability")

seqdplot(biofam.seq, group=bfclara$clustering$cluster3)

## Cluster quality indices estimation using boostrap

bCQI <- bootclustrange(bfclara, biofam.seq, seqdist.args = list(method = "HAM"), 
  R = 3, sample.size = 10,  parallel=FALSE)

bCQI
plot(bCQI, norm="zscore")

\dontrun{
## Fuzzy clustering
bfclaraf <- seqclararange(biofam.seq, R = 3, sample.size = 20, kvals = 2:3, 
  method="fuzzy", seqdist.args = list(method = "HAM"), 
	parallel=FALSE)


bfclaraf
plot(bfclaraf, norm="zscore")


fuzzyseqplot(biofam.seq, group=bfclaraf$clustering$cluster3, type="I", 
  sortv="membership", membership.threashold=0.2)

## Noise clustering
bfclaran <- seqclararange(biofam.seq, R = 3, sample.size = 20, kvals = 2:3, 
  method="noise", seqdist.args = list(method = "HAM"), dnoise=6,
	parallel=FALSE)

fuzzyseqplot(biofam.seq, group=bfclaran$clustering$cluster3, type="I", 
  sortv="membership", membership.threashold=0.2)

}
}

