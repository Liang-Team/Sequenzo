\name{seqemlt}
\alias{seqemlt}
\title{Euclidean Coordinates for Longitudinal Timelines}
\description{
Computes the Euclidean coordinates of sequences from which we get the EMLT  distance between sequences introduced in Rousset et al (2012).
}
\usage{
seqemlt(seqdata, a = 1, b = 1, weighted = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{ a state sequence object defined with the \code{\link[TraMineR]{seqdef}} function.
%%     ~~Describe \code{seqdata} here~~
}
  \item{a}{optional argument for the weighting mechanism that controls the balancing between short term/long term transitions. The weighting function is \eqn{1/(a*s+b)} where \eqn{s} is the transition step.
%%     ~~Describe \code{a} here~~
}
  \item{b}{see argument \code{a}.
%%     ~~Describe \code{b} here~~
}
  \item{weighted}{Logical: Should weights in the sequence object \code{seqdata} be used?
%%     ~~Describe \code{weighted} here~~
}
}
\details{
The EMLT distance is the sum of the dissimilarity between the pairs of states observed at the successive positions, where the dissimilarity between states is defined at each position as the Chi-squared distance between the normalized vectors of transition probabilities (profiles of situations) from the current state to the next observed states in the sequence.
Transition probabilities are down-weighted with the time distance to avoid exaggerated importance of transitions over long periods. The adjustment weight is \eqn{1/a*s+b}, where \eqn{s} is the period length over which the transition probability is measured.

The EMLT distance between two sequences is obtained as the Euclidean distance between the returned numerical sequence coordinates. So, providing \code{coord} as the data input to any clustering algorithm that uses the Euclidean metric is equivalent to cluster with the EMLT metric.

Each time-indexed state is called a situation, and the distance between two states at a position \eqn{t} is derived from the transition probabilities to other observed situations.
%The EMLT distance between sequences takes into account the proximity between situations.
%%A situation may have no occurrence when the referring object is not present during all the duration \eqn{s}.
The distance between any situation and a situation that does not occur is coded as \code{NA}. Such non-occurring situations have no influence on the distance between sequences.
%%  ~~ If necessary, more details than the description above ~~

The obtained numerical representations of sequences may be used as input to any Euclidean algorithm (clustering algorithms, ...).

}
\value{
  An object of class \code{emlt} with the following components:
  \item{coord}{Matrix with in each row the EMLT numerical coordinates of the corresponding sequence.}
  \item{states }{list of states}
  \item{situations }{list of situations (timestamped states)}
  \item{sit.freq }{Situation frequencies}
  \item{sit.transrate }{matrix of transition probabilities from each situation to future situations}
  \item{sit.profil }{profiles of situations. Each profile is the normalized vector of transition probabilities to future situations adjusted to down weight transitions over longer periods.}
  \item{sit.cor }{Matrix of correlations between situations. Two situations are highly correlated when their profiles are similar (i.e., when their transitions towards future are similar). }
}
\references{Rousset, Patrick and  Jean-François Giret (2007), Classifying Qualitative Time Series with SOM: The Typology of Career Paths in France, in F. Sandoval, A. Prieto and M. Grana (Eds) \emph{Computational and Ambient Intelligence}, Lecture Notes in Computer science, vol 4507,  Berlin: Springer, pp 757-764.

Rousset, Patrick and  Jean-François Giret (2008) A longitudinal Analysis of  Labour Market Data with SOM, in J. Rabuñal Dopico, J. Dorado, & A. Pazos (Eds.) \emph{Encyclopedia of Artificial Intelligence}, Hershey, PA: Information Science Reference, pp 1029-1035.

Rousset, Patrick, Jean-François Giret and Yvette Grelet (2012) Typologies De Parcours et Dynamique Longitudinale, \emph{Bulletin de méthodologie sociologique}, 114(1), 5-34.

Studer, Matthias  and Gilbert Ritschard (2014) A comparative review of sequence dissimilarity measures. LIVES Working Paper, 33
\doi{10.12682/lives.2296-1658.2014.33}
%% ~put references to the literature/web site here ~
}
\author{Patrick Rousset and Matthias Studer. Help page by Gilbert Ritschard.
}

\seealso{\code{\link{plot.emlt}}}
\examples{

data(mvad)
mvad.seq <- seqdef(mvad[1:100, 17:41])
alphabet(mvad.seq)
head(labels(mvad.seq))
## Computing distance
mvad.emlt <- seqemlt(mvad.seq)

## typology1 with kmeans in 3 clusters
km <- kmeans(mvad.emlt$coord, 3)

##Plotting by clusters of typology1
seqdplot(mvad.seq, group=km$cluster)

## typology2: 3 clusters by applying hierarchical ward
##   on the centers of the 25 group kmeans solution
km<-kmeans(mvad.emlt$coord, 25)
hc<-hclust(dist(km$centers, method="euclidean"), method="ward")
zz<-cutree(hc, k=3)

##Plotting by clusters of typology2
seqdplot(mvad.seq, group=zz[km$cluster])

## Plotting the evolution of the correlation between states
plot(mvad.emlt, from="employment", to="joblessness", type="cor")
plot(mvad.emlt, from=c("employment","HE", "school", "FE"), to="joblessness", delay=0, leg=TRUE)
plot(mvad.emlt, from="joblessness", to="employment", delay=6)
plot(mvad.emlt, type="pca", cex=0.4, compx=1, compy=2)

}
