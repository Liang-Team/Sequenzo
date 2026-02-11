\name{seqplot.rf}
\alias{seqplot.rf}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Relative Frequency Sequence Plots.
}
\description{
Relative Frequency Sequence Plots (RFS plots) plot a selection of representative sequences as sequence index plots (see \code{\link[TraMineR]{seqIplot}}). RFS plots proceed in several steps. First a set of sequences is ordered according to a substantively meaningful principle, e.g. according to their score on the first factor derived by applying Multidimensional scaling (default) or a user defined sorting variable, such as the timing of a transition of interest. Then the sorted set of sequences is partitioned in to k equal sized frequency groups. For each frequency group the medoid sequence is selected as a representative. The selected representatives are plotted as sequence index plots. RFS plots come with an additional distance-to-medoid box plot that visualizes the distances of all sequences in a frequency group to their respective medoid. Further, an R2 and F-statistic are given that indicate how well the selected medoids represent a given set of sequences. %%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
seqplot.rf(seqdata, k = floor(nrow(seqdata)/10), diss, sortv = NULL,
    ylab=NA, yaxis=FALSE, main=NULL, which.plot="both",
    grp.meth = "first", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{a state sequence object created with the \code{\link[TraMineR]{seqdef}} function.}
  \item{k}{integer: Number of groupings (frequency groups?)}
  \item{diss}{matrix of pairwise dissimilarities between sequences in \code{seqdata} (see \code{\link[TraMineR]{seqdist}}).}
  \item{sortv}{an optional sorting variable that may be used to compute the frequency groups. If \code{NULL}, an MDS is used. Ties are randomly ordered.}
  \item{ylab}{string. An optional label for the y-axis. If set as \code{NA} (default), no label is drawn. Does not apply to \code{which.plot="both"}.}
  \item{yaxis}{logical. Controls whether a y-axis is plotted. When set as \code{TRUE}, the indexes of the sequences are displayed.}
  \item{main}{main graphic title. Default is \code{NULL}.}
  \item{which.plot}{string. One of \code{"both"}, \code{"medoids"}, \code{"diss.to.med"}. When \code{"medoids"}, only the index plot of the medoids is displayed, when \code{"diss.to.med"}, the grouped boxplots of the distances to the medoids is displayed, and when \code{"both"} a combined plot of the two is displayed.}
  \item{grp.meth}{character string. One of \code{"first"} or \code{"random"}. When the number of sequences is not a multiple of the number groups, which groups should have their size augmented by one unit? If \code{"first"}, the first groups, and if \code{"random"} a random selection of groups.}
  \item{\dots}{arguments passed to \code{\link[TraMineR]{seqplot}}.}
 }
\details{
RFS plots are useful to visualize large sets of sequences that cannot be plotted with sequence index plots due to overplotting (see \code{\link[TraMineR]{seqIplot}}). Due to the partitioning into equal sized frequency groups each selected sequence represents an equal portion of the original sample and thereby visually maintains the relative proportion of different types of sequences along the sorting criterion. The ideal number of \eqn{k} fequency groups depends on the size of the original sample and the empirical distribution of the sequences. The larger the sample and the more heterogeneous the sequences, higher numbers of \eqn{k} will be advisable. To avoid overplotting \eqn{k} should generally not be  higher than 200.

Note that distance-to-medoid plots are meaningful only if there are at least 5-10 sequences in each frequency group. The distance-to-medoid plot is not only a quality criterion of how well the medoids represent a respective frequency group. They also provide additional substantive information about how large the variation of sequences is at a given location of the ordered sequences (see Fasang and Liao 2014).

Since ties in \code{sortv} or mds are randomly ordered (see argument \code{ties.method="random"} of function \code{\link{rank}}), one has to set the seed to reproduce exactly the same plot (see \code{\link{set.seed}}).

Unlike other \code{TraMineR} plotting functions, \code{seqplot.rf()} ignores the \code{weights} and does not support the \code{group} argument.
}

\value{
A vector with the group membership (medoid of the group) of each sequence.
}

\references{
Fasang, Anette Eva and Tim F. Liao. 2014. "Visualizing Sequences in the Social Sciences: Relative Frequency Sequence Plots." Sociological Methods & Research 43(4):643-676.
}
\author{
Matthias Studer, Anette Eva Fasang, Tim Liao, and Gilbert Ritschard.
}

\seealso{
See also \code{\link[TraMineR]{seqplot}}, \code{\link[TraMineR]{seqrf}}, \code{\link[TraMineR]{seqrep}}.
}
\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")

## Here, we use only 100 cases selected such that all elements
## of the alphabet be present.
## (More cases and a larger k would be necessary to get a meaningful example.)
biofam.seq <- seqdef(biofam[501:600, ], 10:25, labels=biofam.lab)
diss <- seqdist(biofam.seq, method="LCS")

## Using 12 groups and default MDS sorting
seqplot.rf(biofam.seq, diss=diss, k=12,
   main="Non meaningful example (n=100)")


## With a user specified sorting variable
## Here time spent in parental home: there are ties
## We set a seed because of random order in ties
set.seed(123)
parentTime <- seqistatd(biofam.seq)[, 1]
seqplot.rf(biofam.seq, diss=diss, k=12, sortv=parentTime,
   main="Sorted by parent time")

}

