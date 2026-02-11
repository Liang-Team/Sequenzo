\name{seqsurv}
\alias{seqsurv}
\title{Generate a survfit object for state survival times.}
\description{
  The function considers the spells in the different states in sequences and fits survival curves for each state. Alternatively, for a selected state, it fits the survival curves for each level of a stratifying group variable.
  
  Survival curves are fitted with the \code{\link[survival]{survfit}} function.
}
\usage{
seqsurv(seqdata, groups = NULL, per.state = FALSE, state = NULL,
    with.missing = FALSE)
}
\arguments{
  \item{seqdata}{A sequence \code{stslist} object as defined by the \code{\link[TraMineR]{seqdef}} function.}
  \item{groups}{A stratifying group variable of length equal to the number of sequences.}
  \item{per.state}{Logical. Should the survival probabilites be computed for the state specified as \code{state} argument? If set as \code{TRUE}, the \code{state} argument must also be specified.}
  \item{state}{Single state value or a vector. The short name of the state for which to compute survival probabilities. If a vector of state names, survival probabilities are computed for the subset defined by those states. If \code{NULL}, survival probabilities are computed for all cases.}
  \item{with.missing}{Logical. Should the missing state be accounted for? (Not yet implemented!)}
}
\details{
  The function considers the spells in the different states of a state sequence object (of class \code{stslist}).
  
  When \code{per.state = FALSE}, it fits survival curves for each state in the alphabet.
  Currently, \code{per.state = FALSE} cannot be used with a non-\code{NULL} \code{groups} argument.
  However, \code{\link{seqsplot}} handles this case.
  %%This is done for each level of the \code{groups} variable when it is not \code{NULL}.
   
  When \code{per.state = TRUE}, the survival curve is fitted only for the state provided as \code{state} argument. This is done for each level of the \code{groups} variable.

  Survival curves are fitted with the \code{\link[survival]{survfit}} function.
}
\value{
  An object of class \emph{stslist.surv}. There is a \code{plot} method for such
  objects.
}
%\references{
%  FIXME (Matthias).
%}
%\note{
%  FIXME (Matthias).
%}
\author{Matthias Studer, Gilbert Ritschard, Pierre-Alexandre Fonta}
\seealso{
  \code{\link{plot.stslist.surv}} for basic plots of \emph{stslist.surv} objects
  and \code{\link{seqsplot}} for more elaborated plots.%%(\code{\link{seqplot}} with \code{type = "s"})
  %%for more sophisticated plots allowing grouping and legend.
}
\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
                "Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.short <- c("P","L","M","LM","C","LC","LMC","D")

sple <- 500:700  ## want a sample with all elements of the alphabet
##seqstatl(biofam[sple,10:25])
biofam <- biofam[sple,]

## creating the state sequence object
biofam.seq <- seqdef(biofam[,10:25], alphabet=0:7, states=biofam.short, labels=biofam.lab)

## Spell survival curves
(biofam.surv <- seqsurv(biofam.seq))

## Cohort distinguishing between those born before or after World War II
biofam$wwii <- biofam$birthyr <= 1945

## Separate survival curves in a given state (here LMC "Left+Marr+Child") according to wwii
(biofam.surv <- seqsurv(biofam.seq, groups=biofam$wwii, per.state=TRUE, state="LMC"))
plot(biofam.surv)
}
\keyword{survival}
