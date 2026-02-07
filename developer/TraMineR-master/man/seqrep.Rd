\name{seqrep}
\alias{seqrep}
%%\alias{seqrplot} alias in seqplot
\title{Extracting sets of representative sequences}

\description{
Returns either an as small as possible set of non redundant representatives covering (having in their neighborhood) a desired percentage of all sequences, or a given number of patterns with highest coverage. Special cases are single representatives such as the medoid or the sequence pattern with densest neighborhood. See \code{\link{plot.stslist.rep}} for the plot method and \code{\link{seqplot}} for other plot options.
}

%
%attempts to find an optimal (as small as
%possible for a required coverage) set of representative
%sequences that exhibits the key features of the whole sequence data
%set, the goal being to easily get sounded interpretation of the set of sequences.
%}

\usage{
seqrep(seqdata, criterion = "density", score = NULL, decreasing = TRUE,
  coverage = 0.25, nrep = NULL, pradius = 0.10, dmax = NULL, diss = NULL,
  weighted = TRUE, trep, tsim, dist.matrix, ...)
}

\arguments{
  \item{seqdata}{a state sequence object as defined by the \code{\link{seqdef}} function.}

  \item{criterion}{the representativeness criterion for sorting the candidate list. One of \code{"freq"} (sequence
      frequency), \code{"density"} (neighborhood density), \code{"mscore"} (mean state frequency), \code{"dist"}
      (centrality) and \code{"prob"} (sequence likelihood). See details.}

  \item{score}{an optional vector of representativeness scores for sorting the sequences in the candidate
      list. The length of the vector must be equal to the number of sequences in the sequence object.}

  \item{decreasing}{if a score vector is provided, indicates whether the objects in the candidate list must be sorted in
      ascending or descending order of this score. Default is \code{TRUE}, i.e. descending. The first object in the candidate list
      is then supposed to be the most representative. }

  \item{coverage}{coverage threshold, i.e., minimum proportion of sequences that should have a representative in their
      neighborhood (neighborhood radius is defined by \code{pradius}).}

 \item{nrep}{number of representative sequences. If \code{NULL} (default), the size of the representative set is
     controlled by \code{coverage}.}

  \item{pradius}{neighborhood
radius as a percentage of the maximum (theoretical)
distance \code{dmax}. Defaults to 0.1 (10\%). Sequence \eqn{y} is
redundant to sequence \eqn{x} when it is in the neighborhood of \eqn{x}, i.e., within a distance
\code{pradius*dmax} from \eqn{x}.}

  \item{dmax}{maximum theoretical distance. The \code{dmax} value is used to derive the neighborhood radius as \code{pradius*dmax}. If \code{NULL}, the value of \code{dmax} is derived from the dissimilarity matrix.}

  \item{diss}{matrix of pairwise dissimilarities between sequences in \code{seqdata}. If \code{NULL}, the
      matrix is computed by calling the \code{\link{seqdist}} function. In that case, optional arguments to be passed to
      the \code{seqdist} function (see \code{...} hereafter) should also be provided.}

  \item{weighted}{logical: Should weights assigned to the state sequence object be accounted for? (See \code{\link{seqdef}}.)
  Set as \code{FALSE} to ignore the weights.}
  \item{trep}{Deprecated. Use \code{coverage} instead.}
  \item{tsim}{Deprecated. Use \code{pradius} instead.}
  \item{dist.matrix}{Deprecated. Use \code{diss} instead.}
  \item{...}{optional arguments to be passed to the \code{seqdist} function, mainly \code{dist.method} specifying the
      metric for computing the distance matrix, \code{norm} for normalizing the distances, \code{indel} and \code{sm} for
      indel and substitution costs when Optimal Matching metric is chosen. See \code{\link{seqdist}} manual page for
      details.}
}

\details{
 The representative set is obtained by an heuristic. Representatives are selected by successively extracting from  the sequences sorted by their representativeness score those which are not redundant with already retained representatives. The selection stops when either the desired coverage or the wanted number of representatives is reached. Sequences are sorted either by the values provided as \code{score} argument or by specifying one of the following as \code{criterion} argument: \code{"freq"} (\emph{sequence frequency}), \code{"density"}
(\emph{neighborhood density}), \code{"mscore"} (\emph{mean state frequency}), \code{"dist"}
(\emph{centrality}), and \code{"prob"} (\emph{sequence likelihood}).

With the \emph{sequence frequency} criterion, the more frequent a
sequence the more representative it is supposed to be. Therefore, sequences are sorted in decreasing frequency order.

The \emph{neighborhood density} is the
number---density---of sequences in the neighborhood of the
sequence. This requires to set the neighborhood radius
\code{pradius}.  Sequences are
sorted in decreasing density order.

The \emph{mean state frequency} criterion is the mean value of the transversal frequencies of the successive states.
Let \eqn{s=s_{1}s_{2}\cdots s_{\ell}}{s=(s_1, s_2, ..., s_l)} be a sequence of length \eqn{\ell}{l} and \eqn{(f_{s_1},
f_{s_2}, \ldots, f_{s_\ell})}{f(s_1), f(s_2), ..., f(s_l)} the frequencies of the states at (time-)position \eqn{(t_1,
t_2,\ldots t_{\ell})}{t_1, t_2, ..., t_l}. The mean state frequency is the sum of the state frequencies divided by the
sequence length
\deqn{
MSF(s)=\frac{1}{\ell} \sum_{i=1}^{\ell} f_{s_{i}}
}{
	MSF(s) = sum f(s_i) / l
}
%
The lower and upper boundaries of \eqn{MSF} are \eqn{0} and \eqn{1}. \eqn{MSF} is equal to \eqn{1} when all the sequences
in the set are identical, i.e. when there is a single sequence pattern. The most representative sequence is the one with
the highest score.

The \emph{centrality} criterion is the sum of distances to all other sequences. The
smallest the sum, the most representative is the sequence.

The \emph{sequence likelihood} \eqn{P(s)} is defined as the product of the probability with which each of its observed
successive state is supposed to occur at its position.
Let  \eqn{s=s_{1}s_{2} \cdots s_{\ell}}{s_1, s_2, s_l}  be a sequence of length \eqn{\ell}{l}. Then
\deqn{
P(s)=P(s_{1},1) \cdot P(s_{2},2) \cdots P(s_{\ell},\ell)
}{
P(s)=P(s_1,1) * P(s_2,2) * ... * P(s_l,l)
}
with \eqn{P(s_{t},t)}{P(s_t,t)} the probability to observe state \eqn{s_t} at position \eqn{t}.
\cr
The question is how to determinate the state probabilities \eqn{P(s_{t},t)}{P(s_t,t)}. One commonly used method for
computing them is to postulate a Markov Chain model, which can be of various order. The implemented criterion considers the
probabilities derived from the first order Markov model, that is each \eqn{P(s_{t},t)}{P(s_t,t)}, \eqn{t>1} is set to the
transition rate \eqn{p(s_t|s_{t-1})}{p(s_t|s_(t-1))} estimated across sequences from the observations at positions \eqn{t}
and \eqn{t-1}. For \eqn{t=1}, we set \eqn{P(s_1,1)} to the observed frequency of the state \eqn{s_1} at position 1.

The likelihood \eqn{P(s)} being generally very small, we use
\eqn{-\log P(s)}{-log P(s)} as sorting criterion. The latter quantity reaches its minimum for
\eqn{P(s)} equal to 1, which leads to sort the sequences in
ascending order of their score.

Use \code{criterion="dist"} (centrality) and \code{nrep=1} to get the medoid, and \code{criterion="density"} and \code{nrep=1} to get the densest sequence pattern.

For more details, see \cite{Gabadinho & Ritschard, 2013}.
}

\value{An object of class \code{stslist.rep}. This is actually a
state sequence object (containing a list of state sequences) with the
following additional attributes:
  \item{Scores}{a vector with the representative score of each sequence in the original set given the chosen criterion.}
  \item{Distances}{a matrix with the distance of each sequence to its nearest representative.}
  \item{Rep.group}{vector with, for each sequence, the representative that represents it.}
  \item{idx.rep}{list with indexes of occurrences of each representative in original data.}
  \item{Statistics}{a data frame with quality measures for each representative sequence: number \eqn{na} of sequences
      attributed to the representative, number \eqn{nb} of sequences in the representative's neighborhood, mean distance \eqn{MD} to the
      representative and a few other indexes.}
  \item{Quality}{overall quality measure.}

Print, plot and summary methods are available. More elaborated plots
are produced by the \code{\link{seqplot}} function using the \code{type="r"}
argument, or the \code{\link{seqrplot}} alias. }

\references{
{Gabadinho A, Ritschard G (2013). "Searching for typical life trajectories applied to child birth histories", In R Lévy, E. Widmer (eds.), \emph{Gendered Life Courses}, pp. 287-312. Vienna: LIT.}


{Gabadinho A, Ritschard G, Studer M, Müller NS (2011). "Extracting and Rendering Representative Sequences", In A Fred, JLG Dietz, K Liu, J Filipe (eds.), \emph{Knowledge Discovery, Knowledge Engineering and Knowledge Management}, volume 128 of \emph{Communications in Computer and Information Science (CCIS)}, pp. 94-106. Springer-Verlag.}
}

\seealso{\code{\link{seqplot}}, \code{\link{plot.stslist.rep}}, \code{\link{dissrep}}, \code{\link{disscenter}}}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam[,10:25], labels=biofam.lab)

## Computing the distance matrix
costs <- seqsubm(biofam.seq, method="TRATE")
biofam.om <- seqdist(biofam.seq, method="OM", sm=costs)

## Representative set using the neighborhood density criterion
biofam.rep <- seqrep(biofam.seq, diss=biofam.om, criterion="density")
biofam.rep
summary(biofam.rep)
plot(biofam.rep)

## plot by groups represented by the representatives
seqdplot(biofam.seq, group=attr(biofam.rep,"Rep.group"), border=NA)

## indexes of sequences represented by 1st representative
r1.grp <- which(attr(biofam.rep,"Rep.group")==1)
## indexes of occurrences of the first representative sequence
attr(biofam.rep,"idx.rep")[[1]]
}

\author{Alexis Gabadinho and Gilbert Ritschard}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
