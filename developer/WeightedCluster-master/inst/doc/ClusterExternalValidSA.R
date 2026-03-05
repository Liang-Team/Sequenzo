## ----include=FALSE------------------------------------------------------------
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
mvad.seq <- seqdef(mvad, 17:86, alphabet = mvad.alphabet, states = mvad.shortlab, labels = mvad.lab, xtstep = 6)


## ----cache=TRUE, message=FALSE------------------------------------------------
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

## -----------------------------------------------------------------------------
cla <- clustassoc(clustqual, diss=diss, covar=mvad$funemp)
cla

## -----------------------------------------------------------------------------
plot(cla, main="Unaccounted")

## ----fig.width=8, fig.height=8------------------------------------------------
seqdplot(mvad.seq, group=clustqual$clustering$cluster5, border=NA)

## ----fig.width=8, fig.height=8------------------------------------------------
seqdplot(mvad.seq, group=clustqual$clustering$cluster6, border=NA)

## ----include=FALSE------------------------------------------------------------
knitr::write_bib(file = 'packages.bib')

