## ----setup, include=FALSE-----------------------------------------------------
knitr::knit_hooks$set(time_it = local({
  now <- NULL
  function(before, options) {
    if (before) {
      # record the current time before each chunk
      now <<- Sys.time()
    } else {
      # calculate the time difference after a chunk
      res <- difftime(Sys.time(), now, units = "secs")
      # return a character string to show the time
      paste("Time for this code chunk to run:", round(res,
        2), "seconds")
    }
  }
}))
knitr::opts_chunk$set(dev = "png", dev.args = list(type = "cairo-png"), time_it=TRUE)

## ----message=FALSE------------------------------------------------------------
set.seed(1)

## ----message=FALSE------------------------------------------------------------
## Loading the TraMineR library
library(TraMineR)
## Loading the data
data(mvad)

## State properties
mvad.alphabet <- c("employment", "FE", "HE", "joblessness", "school", "training")
mvad.lab <- c("employment", "further education", "higher education", "joblessness", "school", "training")
mvad.shortlab <- c("EM","FE","HE","JL","SC","TR")

## Creating the state sequence object
mvad.seq <- seqdef(mvad[, 17:86], alphabet = mvad.alphabet, states = mvad.shortlab, 
                   labels = mvad.lab, xtstep = 6)

## ----message=FALSE------------------------------------------------------------
## Using fastcluster for hierarchical clustering
library(fastcluster)
## Distance computation
diss <- seqdist(mvad.seq, method="LCS")
## Hierarchical clustering
hc <- hclust(as.dist(diss), method="ward.D")

## ----message=FALSE------------------------------------------------------------
# Loading the WeightedCluster library
library(WeightedCluster)
# Computing cluster quality measures.
clustqual <- as.clustrange(hc, diss=diss, ncluster=10)
clustqual

## ----fig.width=8, fig.height=10-----------------------------------------------
seqdplot(mvad.seq, group=clustqual$clustering$cluster6, border=NA)

## ----message=FALSE------------------------------------------------------------
# Add the clustering solution to the dataset
mvad$clustering <- clustqual$clustering$cluster6
# The first argument is a formula for the association between clustering and covariates of interest
# The function estimates separate logistic regressions for each cluster and compute the corresponding AMEs
# with their confidence interval
rarcatout <- rarcat(clustering ~ funemp + gcse5eq, data=mvad, diss=diss, robust=FALSE)
rarcatout

## ----message=FALSE------------------------------------------------------------
# # Loading the parallel library
library(future)
plan(multisession)

## ----message=FALSE------------------------------------------------------------
# Evaluate the validity of the original analysis and the reliability of its
# findings by applying RARCAT with 50 bootstrap replications.
# As in the original analysis, hierarchical clustering with Ward method is implemented.
# The number of clusters is fixed to 6 here.
rarcatout <- rarcat(clustering ~ funemp + gcse5eq, data = mvad, diss = diss, robust = TRUE, R = 100, 
                    kmedoid = FALSE, hclust.method = "ward.D", 
                    fixed = TRUE, ncluster = 6)
rarcatout

## ----message=FALSE------------------------------------------------------------
# Bootstrap replicates of the typology and its association with the variable funemp.
# As in the original analysis, hierarchical clustering with Ward method is implemented.
# Also, an optimal clustering solution with n between 2 and 10 is evaluated each time by
# maximizing the HC index.

# As we previously initalized parallel computing, it is used in these computations
vary.rarcat <- rarcat(clustering ~ funemp, data = mvad, diss = diss, R = 50,
                       kmedoid=FALSE, hclust.method = "ward.D",
                       fixed = FALSE, ncluster = 10, cqi = "HC")
vary.rarcat

## ----fig.width=8, fig.height=10-----------------------------------------------
# Histogram of the estimated AMEs for all individuals and all bootstraps
plot(rarcatout, covar="funempyes")

## ----message=FALSE------------------------------------------------------------
summary(rarcatout)

## ----message=FALSE, fig.width=8, fig.height=10--------------------------------
plot(rarcatout, what="ranef", covar="funempyes")

## ----message=FALSE, fig.width=8, fig.height=10--------------------------------
outliers <- abs(rarcatout$observation.stdranef[, "funempyes"])>2
seqIplot(mvad.seq[outliers, ], group=mvad$clustering[outliers])

## ----message=FALSE------------------------------------------------------------
# Loading the fpc library
library(fpc)
# Cluster-wise stability assessment by bootstrap
stab <- clusterboot(diss, B = 500, distances = TRUE, clustermethod = disthclustCBI, 
                    method = "ward.D", k = 6, count = FALSE)
stab

## ----include=FALSE------------------------------------------------------------
plan(sequential)

