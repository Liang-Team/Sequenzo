\encoding{latin1}
\name{group.p}
\alias{group.p}
\author{Gilbert Ritschard}

\title{Adds proportion of occurrences to each level names}


\description{
Adds the proportion of occurrences of each level to the corresponding level name.}

\usage{
group.p(group, weights=NULL)
}

\arguments{
  \item{group}{A group variable.}
  
  \item{weights}{Vector of weights of same length as the group variable.} 
}

\details{
    The \code{group} variable can be a factor or a numerical variable. In the latter case it is transformed to a factor.
}

\seealso{
\code{\link[TraMineR]{seqplot}}.
}

\examples{
data(actcal)
actcal <- actcal[1:100,]
actcal.seq <- seqdef(actcal[,13:24])
seqdplot(actcal.seq, group=group.p(actcal$sex))

levels(group.p(actcal$sex, weights=runif(length(actcal$sex))))
}
\keyword{util}
