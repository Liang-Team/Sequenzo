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
diss <- seqdist(mvad.seq, method="HAM")
## Hierarchical clustering
hc <- hclust(as.dist(diss), method="ward.D")

## ----message=FALSE------------------------------------------------------------
# Loading the WeightedCluster library
library(WeightedCluster)
# Computing cluster quality measures.
clustqual <- as.clustrange(hc, diss=diss, ncluster=10)
clustqual

## -----------------------------------------------------------------------------
bcq.combined <- seqnullcqi(mvad.seq, clustqual, R=50, model="combined", seqdist.args=list(method="HAM"), hclust.method="ward.D", parallel = TRUE)

## ----size="tiny"--------------------------------------------------------------
bcq.combined

## ----size="tiny"--------------------------------------------------------------
print(bcq.combined, norm=FALSE)

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.combined, type="seqdplot")

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.combined, stat="ASW", type="density")

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.combined, stat="ASW", type="line")

## -----------------------------------------------------------------------------
bcq.seq <- seqnullcqi(mvad.seq, clustqual, R=50, model="sequencing", seqdist.args=list(method="HAM"), hclust.method="ward.D", parallel = TRUE)

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.seq, stat="ASW", type="line")

## -----------------------------------------------------------------------------
bcq.dur <- seqnullcqi(mvad.seq, clustqual, R=50, model="duration", seqdist.args=list(method="HAM"), hclust.method="ward.D", parallel = TRUE)

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.dur, stat="ASW", type="line")

## ----cache=TRUE---------------------------------------------------------------
bcq.stateindep <- seqnullcqi(mvad.seq, clustqual, R=50, model="stateindep", seqdist.args=list(method="HAM"), hclust.method="ward.D", parallel = TRUE)

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.stateindep, stat="ASW", type="line")

## ----cache=TRUE---------------------------------------------------------------
bcq.Markov <- seqnullcqi(mvad.seq, clustqual, R=50, model="Markov", seqdist.args=list(method="HAM"), hclust.method="ward.D", parallel = TRUE)

## ----fig.width=8, fig.height=5, results="hide", dev="png"---------------------
plot(bcq.Markov, stat="ASW", type="line")

## ----fig.width=10, fig.height=12, results="hide"------------------------------
seqdplot(mvad.seq, clustqual$clustering$cluster9, border=NA)

