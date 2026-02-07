\name{dissrep}
\alias{dissrep}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Extracting sets of representative objects using a dissimilarity matrix}
\description{
The function extracts a set of representative objects that exhibits the key features of the whole data set, the goal being to get easy sounded interpretation of the latter. The user can set either the desired coverage level (the proportion of objects having a representative in their neighborhood) or the desired number of representatives.
}
\usage{
dissrep(diss, criterion = "density", score = NULL, decreasing = TRUE,
  coverage = 0.25, nrep = NULL, pradius = 0.10, dmax = NULL,
  weights = NULL, trep, tsim)
}
%- maybe also 'usage' for other objects documented here.

\arguments{
 \item{diss}{A dissimilarity matrix or a \code{dist} object (see \code{\link{dist}})}

  \item{criterion}{the representativeness criterion for sorting the
candidate list. One of \code{"freq"} (frequency), \code{"density"}
(neighborhood density) or \code{"dist"} (centrality). An optional
vector containing the scores for sorting the candidate objects may
also be provided. See below and details.}

  \item{score}{an optional vector containing the representativeness
scores used for sorting the objects in the candidate list. The length
of the vector must be equal to the number of rows/columns in the
distance matrix, i.e the number of objects.}

  \item{decreasing}{logical. If a score vector is provided, should
the objects in the candidate list be sorted in ascending order of the score. If \code{FALSE}, sort is in descending order. The first object in the candidate list is supposed to be the most representative.}

  \item{coverage}{controls the size of the representative set by setting
the desired coverage level, i.e the proportion of objects having a
representative in their neighborhood. Neighborhood radius is defined
by \code{pradius}.}

  \item{nrep}{number of representatives. If \code{NULL} (default),
\code{coverage} argument is used to control the size of the
representative set.}

  \item{pradius}{neighborhood
radius as a percentage of the maximum (theoretical)
distance  \code{dmax}. Defaults to 0.1 (10\%). Object \eqn{y} is
redundant to object \eqn{x} when it is in the neighborhood of \eqn{x}, i.e., within a distance
\code{pradius*dmax} from \eqn{x}.}

  \item{dmax}{maximum theoretical distance. The \code{dmax} value is used to derive the neighborhood radius as \code{pradius*dmax}. If \code{NULL}, the value of \code{dmax} is derived from the dissimilarity matrix.}

  \item{weights}{vector of weights of length equal to the number of rows of the dissimilarity matrix. If \code{NULL}, equal weights are assigned.}
  \item{trep}{Deprecated. Use \code{coverage} instead.}
  \item{tsim}{Deprecated. Use \code{pradius} instead.}
 }

\details{
 The representative set is obtained by an heuristic. Representatives are selected by successively extracting from  the sequences sorted by their representativeness score those which are not redundant with already retained representatives. The selection stops when either the desired coverage or the wanted number of representatives is reached. Objects are sorted either by the values provided as \code{score} argument, or by specifying one of the following as \code{criterion} argument: \code{"freq"} (\emph{sequence frequency}), \code{"density"}
(\emph{neighborhood density}), \code{"dist"}
(\emph{centrality}).

The \emph{frequency} criterion uses the frequencies as
representativeness score. The frequency of an object in the data is
computed as the number of other objects with whom the dissimilarity
is equal to 0. The more frequent an object the more representative it
is supposed to be. Hence, objects are sorted in decreasing frequency
order. This criterion is equivalent to the neighborhood (see below)
criterion with a neighborhood radius equal to 0.

The \emph{neighborhood density} is the
number---density---of objects in the neighborhood of the object. This requires to set the neighborhood radius \code{pradius}.  Objects are
sorted in decreasing density order.


The \emph{centrality} criterion is the sum of distances to all other objects. The
smallest the sum, the most representative the object.

Use \code{criterion="dist"} (centrality) and \code{nrep=1} to get the medoid and \code{criterion="density"} and \code{nrep=1} to get the densest object pattern.


For more details, see \cite{Gabadinho and Ritschard, 2013}.
}

\value{An object of class \code{diss.rep}. This is a vector containing
the indexes of the representative objects with the following additional attributes:
  \item{Scores}{vector with the representative score of each object given the chosen criterion.}
  \item{Distances}{matrix with the distance of each object to its nearest representative.}
  \item{Rep.group}{vector with, for each object, the representative that represents it.}
  \item{idx.rep}{list with indexes of occurrences of each representative in original data.}
  \item{Statistics}{a data frame with quality measures for each representative: number of objects assigned to the representative,
number of objects in the representative's neighborhood, mean distance
to the representative.}
  \item{Quality}{overall quality measure.}

Print and summary methods are available.
}

\references{
{Gabadinho A, Ritschard G (2013). "Searching for typical life trajectories applied to child birth histories", In R Lévy, E. Widmer (eds.), \emph{Gendered Life Courses}, pp. 287-312. Vienna: LIT.}

Gabadinho A, Ritschard G, Studer M, Müller NS (2011). "Extracting and Rendering Representative Sequences", In A Fred, JLG Dietz, K Liu, J Filipe (eds.), \emph{Knowledge Discovery, Knowledge Engineering and Knowledge Management}, volume 128 of \emph{Communications in Computer and Information Science (CCIS)}, pp. 94-106. Springer-Verlag.
}

\seealso{\code{\link{seqrep}}, \code{\link{disscenter}}}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
"Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam[, 10:25], labels=biofam.lab)

## Computing the distance matrix
costs <- seqsubm(biofam.seq, method="TRATE")
biofam.om <- seqdist(biofam.seq, method="OM", sm=costs)

## Representative set using the neighborhood density criterion
biofam.rep <- dissrep(biofam.om)
biofam.rep
summary(biofam.rep)
## indexes of first occurrence of second representatives in original data
attr(biofam.rep,"idx.rep")[[2]][1]

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.

\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Dissimilarity-based analysis}
