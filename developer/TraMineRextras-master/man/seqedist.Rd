\encoding{latin1}
\name{seqedist}
\alias{seqedist}
\title{Distances between event sequences}
\description{
Compute Optimal-Matching-like distances between event sequences. The distance measure is described in \cite{Studer et al. 2010}.
}
\usage{
seqedist(seqe, idcost, vparam, interval="No", norm="YujianBo")
}
\arguments{
  \item{seqe}{An event sequence \code{seqelist} object as defined by the \code{\link[TraMineR]{seqecreate}} function.}
  \item{idcost}{Insertion/deletion cost of the different events (one entry per element of the event alphabet).}
  \item{vparam}{Positive real. The cost for a one-unit change in the time stamp of an event.}
  \item{norm}{Character. One of \code{"YujianBo"} (respects triangle inequality), \code{"max"} (maximum distance) or \code{"none"}.}
  \item{interval}{Character. One of \code{"No"} (absolute ages), \code{"previous"} (time spent since previous event) or \code{"next"} (time spent until next event).}
}
\value{a distance matrix.}

\author{Matthias Studer}

\references{
Studer, M., Müller, N.S., Ritschard, G. & Gabadinho, A. (2010), "Classer, discriminer et visualiser des séquences d'événements",
In Extraction et gestion des connaissances (EGC 2010), \emph{Revue des nouvelles technologies de l'information RNTI}. Vol. E-19, pp. 37-48.

Ritschard, G., Bürgin, R., and Studer, M. (2014), "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) \emph{Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences}. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.

}

\examples{
data(actcal.tse)
actcal.seqe <- seqecreate(actcal.tse[1:200,])[1:6,]
## We have 8 different events in this dataset
idcost <- rep(1, 8)
dd <- seqedist(actcal.seqe, idcost=idcost, vparam=.1)
}

\keyword{misc}

