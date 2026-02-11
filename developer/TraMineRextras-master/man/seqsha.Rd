\name{seqsha}
\alias{seqsha}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Sequence History Analysis (SHA)
}
\description{
Sequence History Analysis (SHA) aims to study how a previous trajectory is linked to an upcoming event. This procedure relies on sequence analysis typologies to identify the type of previous trajectory as a time-varying covariate and uses discrete-time survival models to estimate its relationship with the upcoming event under consideration.
}
\usage{
seqsha(seqdata, time, event, include.present = FALSE, align.end = FALSE, covar = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{seqdata}{State sequence object created with the \code{\link[TraMineR]{seqdef}} function. The whole trajectory followed by individuals.}
  \item{time}{Numeric. The time of occurrence of the event or the observation time for censored observations.}
  \item{event}{Logical. Whether the event occured or not (censored observations).}
  \item{include.present}{Logical. If \code{FALSE} (default), the state occurring at the current time is not included in the previous trajectory.}
  \item{align.end}{Logical. If \code{FALSE} (default), the previous trajectories are aligned at the beginning of the trajectory. If \code{TRUE}, the previous trajectories are aligned on the end.}
\item{covar}{Optional \code{data.frame} storing covariates of interest. These covariates are added to the final data set. }

}
\details{
SHA works in four steps. First, it makes use of a discrete-time representation of the data also known as person-period file. In this format, one observation is generated for each individual  at each time point. Second, the previous trajectory at each time point is coded as the sequence of states from the beginning (t=1 in our example) until the previous position t-1. Third, a typology of the previous trajectory is created using standard sequence analysis procedure. This results in a new time-varying covariate coding the type of previous trajectory at each time point. Fourth, the relationship between the previous trajectory and the subsequent event is estimated using a discrete-time model, which includes the past trajectory type as a covariate. In this step, other covariates can be included as well.

The \code{seqsha} function can be used to automatically reorganize the data according to the first two steps described above. Then, a standard procedure can be applied on the resulting data set. The example section below provides an example of the whole procedure.

}
\value{
A data frame with the following variables:
\item{id}{Numeric. The ID of the observation as the row number in the original \code{seqdata}.}
\item{time}{Numeric. The time unit from the beginning of the original sequence until the occurence of the event.}
\item{event}{Logical. Whether the event occured within this time unit.}
\item{T1 until T...}{The state sequence coding the previous trajectory. Columns names depends on \code{seqdata} and \code{aligne.end}.}
\item{Optional covariate list}{The covariates provided with the \code{covar} argument.}
}
\seealso{\code{\link{seqcta}}, \code{\link{seqsamm}}
}


\references{
Rossignon F., Studer M., Gauthier JA., Le Goff JM. (2018). Sequence History Analysis (SHA): Estimating the Effect of Past Trajectories on an Upcoming Event. In: Ritschard G., Studer M. (eds) \emph{Sequence Analysis and Related Approaches}. Life Course Research and Social Policies, vol 10. Springer: Cham. \doi{10.1007/978-3-319-95420-2_6}
}
\author{
Matthias Studer
}

\examples{

## Create seq object for biofam data.
data(biofam)
## Reduce the biofam data to accelerate example
biofam <- biofam[100:300,]
bf.shortlab <- c("P","L","M","LM","C","LC", "LMC", "D")
bf.seq <- seqdef(biofam[,10:25], states=bf.shortlab)

## We focus on the occurrence the start of a LMC spell
## The code below aims to find when this event occurred (and whether it occurred).
bf.seq2 <- seqrecode(bf.seq, recodes=list(LMC="LMC"), otherwise = "Other")
dss <- seqdss(bf.seq2)
## Time until LMC spell
time <- ifelse(dss[, 1]=="LMC", 1, seqdur(bf.seq2)[, 1])
## Whether the event (start of LMC spell) started or not
event <- dss[, 1]=="LMC"|dss[, 2]=="LMC"

## The seqsha function will convert the data to person period.
## At each time point, the previous trajectory until that point is stored
sha <- seqsha(bf.seq, time, event, covar=biofam[, c("sex", "birthyr")])
summary(sha)

\dontrun{
## Now we build a sequence object for the previous trajectory
previousTraj <- seqdef(sha[, 4:19])
seqdplot(previousTraj)
## Now we cluster the previous trajectories
##Compute distances using only the dss
## Ensure high sensitivity to ordering of the states
diss <- seqdist(seqdss(previousTraj), method="LCS")
##Clustering with pam
library(cluster)
pclust <- pam(diss, diss=TRUE, k=4, cluster.only=TRUE)
#Naming the clusters
sha$pclustname <- factor(paste("Type", pclust))
##Plotting the clusters to make senses of them.
seqdplot(previousTraj, sha$pclustname)


## Now we use a discrete time model include the type of previous trajectory as covariate.
summary(glm(event~time+pclustname+sex, data=sha, family=binomial))
}

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
