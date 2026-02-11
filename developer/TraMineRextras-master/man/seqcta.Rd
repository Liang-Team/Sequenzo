\name{seqcta}
\alias{seqcta}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Competing Trajectory Analysis (CTA)
}
\description{
Competing Trajectory Analysis (CTA) aims to simultaneously study the occurrence of an event and the trajectory following it over a pre-defined period of time. The \code{seqcta} function convert the data to run the analysis.
}
\usage{
seqcta(seqdata, subseq = 5, time = NULL, event = NULL, initial.state = NULL, covar = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{State sequence object created with the \code{\link[TraMineR]{seqdef}} function. The whole trajectory followed by individuals.}
  \item{subseq}{Numeric. The length of the trajectory following the event to be considered.}
  \item{time}{Numeric. The time of occurrence of the event, can be \code{NA} for censored observations.  If \code{NULL} (default), \code{initial.state} should be provided.}
  \item{event}{Logical. Whether the event occur for each trajectory. If \code{NULL} (default) and \code{time} is provided, \code{NA} \code{time} values are used to detect censored observations.}
  \item{initial.state}{Character. Only used if \code{time} is not provided. If provided, the end of the first spell of the sequence, but only  in \code{initial.state} state, is used as the event of interest to compute the \code{time} argument.}
  \item{covar}{Optional \code{data.frame} storing covariates of interest. These covariates are added to the final data set. }
}
\details{
Competing Trajectory Analysis (CTA) works as follows. First, the sequence following the studied event are clustered. Second, the type of trajectory followed is linked with covariates using a competing risks model.

The \code{seqcta} function reorganizes the data to run CTA. More precisely, it provides a person-period data frame until the occurrence of the event. When the event occurs, the trajectory following it is also stored. Covariates specified using the \code{covar} arguments are also stored.

The example section below provides a step by step example of the whole procedure.

}
\value{
A data frame with the following variables
\item{id}{Numeric. The ID of the observation as the row number in the original \code{seqdata}.}
\item{time}{Numeric. The time unit from the beginning of the original sequence until the occurence of the event.}
\item{event}{Logical. Whether the event occured within this time unit.}
\item{lastobs}{Logical. Whether this is the last observation for an individual observation, censored or not. This is useful when one want only one row per individual, for instance to plot survival curves (see example).}
\item{T1 until T...}{The state sequence following the event starting from 1 (time unit after the event) until \code{subseq} time units after the event. Only available for the rows where \code{event=TRUE}.}
\item{Optional covariate list}{The covariates provided with the \code{covar} argument.}
}

\seealso{\code{\link{seqsamm}}, \code{\link{seqsha}}
}

\references{
M. Studer, A. C. Liefbroer and J. E. Mooyaart, 2018. Understanding trends in family formation trajectories: An application of Competing Trajectories Analysis (CTA), \emph{Advances in Life Course Research} 36, pp 1-12.  \doi{10.1016/j.alcr.2018.02.003}
}
\author{
Matthias Studer
}
\examples{
## Create seq object for biofam data.
data(biofam)
bf.shortlab <- c("P","L","M","LM","C","LC", "LMC", "D")
bf.seq <- seqdef(biofam[,10:25], states=bf.shortlab)

## We focus on the occurrence of ending the first "P" spell and the trajectory that follows
## For the next subseq=5 years
## We also store the covariate sex and birthyr
## seqcta will transform the data to person-period until the end of the first "P" spell
## and store the following trajectory

cta <- seqcta(bf.seq, subseq=5, initial.state="P", covar=biofam[, c("sex", "birthyr")])
summary(cta)

## If the studied event is not a first state of the trajectory
## One can also provide the event using the time and event arguments
## Here we compute the time spent in "P" ourselves before providing it to seqcta

dur <- seqdur(bf.seq)
## If "P" is the first state, we use the time in this state (dur[, 1])
## Otherwise we use 0 (started immediatly at the beginning)
timeP <- ifelse(bf.seq[, 1]=="P", dur[, 1], 0)

## The event occured if timeP is inferior to the length of the sequence
## Otherwise they never left their parents.
eventP <- timeP < 16

cta2 <- seqcta(bf.seq, subseq=5, time=timeP, event=eventP, covar=biofam[, c("sex", "birthyr")])
##Identical results
summary(cta2)


## Not run to save computation time
\dontrun{
library(survival)

## To plot a survival curve, we only need the last observation for each individual.
## Kaplan Meier curve for the occurrence of the event
ss <- survfit(Surv(time, event)~sex, data=cta, subset=lastobs)
plot(ss, col=1:2)

## Now we cluster the trajectories following the event
## Therefore we only keep lines where the event occured.
clusterTraj <- seqdef(cta[cta$event, 5:9])
##Compute distances
diss <- seqdist(clusterTraj, method="HAM")
##Clustering with pam
library(cluster)
pclust <- pam(diss, diss=TRUE, k=5, cluster.only=TRUE)
#Naming the clusters
pclustname <- paste("Type", pclust)
##Plotting the clusters to make senses of them.
seqdplot(clusterTraj, pclustname)

##Now we store back the clustering in the original person-period data
## We start by adding a variable storing "no event" for all lines
cta$traj.event <- "No event"
## Then we store the type of following trajectory
## only for those having experienced the event
cta$traj.event[cta$event] <- pclustname


## Checking the results
summary(cta)

## Now we can estimate a competing risk model
## Several strategies are available.
## Here we use multinomial model on the person period.

library(mlogit)
summary(mlogit(traj.event~1|time+sex, data=cta, shape="wide", reflevel="No event"))
library(nnet)
summary(multinom(traj.event~time+sex+scale(birthyr), data=cta))

## The model can also be estimated with cox regression
## However, we need to estimate one model for each competing risk
## ie. the type of following trajectory in our case.

## Compute the event variable for "Type 1"

cta$eventType1 <- cta$traj.event=="Type 1"
summary(coxph(Surv(time, eventType1)~sex+scale(birthyr), data=cta, subset=lastobs))


}
}

