\name{plot.emlt}
\alias{plot.emlt}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Emlt Plots
%%  ~~function to do ... ~~
}
\description{Plots static and dynamic state structure from the outcome of \code{seqemlt}. Two types of plot are proposed: The evolution in time of the correlation between states, and the projection of situations (time-indexed states) on their principal planes.
%%  Generic function for
}
\usage{
\method{plot}{emlt}(x, from, to, delay=NULL, leg=TRUE, type="cor", cex=0.7, compx=1, compy=2, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{an object of class \code{emlt} as produced by \code{\link{seqemlt}}
}
  \item{type}{character string: type of plot to be drawn. Possible types are
\code{"cor"} for the evolution in time of the correlation between states, and
\code{"pca"} for the projection of states/situations on their principal planes
}
  \item{from}{vector of state labels: for type \code{"cor"}, origin state(s) to be considered.
}
  \item{to}{state label: for type "cor", destination state.
}
  \item{delay}{for type "cor", the delay (number of time periods) between \code{"from"} and \code{"to"} arguments. The correlation between state \code{"from"} at time t and \code{"to"} at t+\code{delay}. By default \code{delay} is 0.
}
  \item{compx}{integer: for type \code{"pca"} first component, axis x
}
  \item{compy}{integer: for type \code{"pca"} second component, axis y
}
\item{leg}{logical: should the legend be included}
\item{cex}{numerical value: amount by which plotting text and symbols should be magnified relative to the default.}
  \item{\dots}{Arguments to be passed to methods, such as graphical parameters (see \code{\link{par}})
}
}
\details{The evolution of the correlation reveals the evolution of the emlt Euclidean distance between the situations (time-indexed states) along the timeframe.

The \code{"pca"} components are the principal components of the emlt numerical coordinates of the sequences, see \code{\link{seqemlt}}.
%%  ~~ If necessary, more details than the description above ~~
}
%\references{
%}
\author{Patrick Rousset and Matthias Studer
}

\seealso{
See also \code{\link{seqemlt}} (with examples)
}
\examples{
## See examples on 'seqemlt' help page
}
