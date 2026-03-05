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
library(WeightedCluster)

## ----seqdefbiofam, warning=FALSE, message=FALSE, fig.width=8, fig.height=5----
data(biofam) #load illustrative data
## Defining the new state labels 
statelab <- c("Parent", "Left", "Married", "Left/Married",  "Child", 
            "Left/Child", "Left/Married/Child", "Divorced")
## Creating the state sequence object,
biofam.seq <- seqdef(biofam[,10:25], alphabet=0:7, states=statelab)
seqdplot(biofam.seq, legend.prop=0.2)

## ----seqclaraex, warning=FALSE, message=FALSE---------------------------------
bfclara <- seqclararange(biofam.seq, R = 50, sample.size = 100, kvals = 2:10, 
						 seqdist.args = list(method = "HAM"), parallel=TRUE, 
                         stability=TRUE)

## ----plotcqi, fig.width=8, fig.height=5---------------------------------------
bfclara
plot(bfclara, norm="range")

## ----plotcqistabilityavg, fig.width=8, fig.height=5---------------------------
plot(bfclara, stat="stabmean")

## ----plotcqistability, fig.width=8, fig.height=5------------------------------
plot(bfclara, stat="stability")

## ----bcqi, fig.width=8, fig.height=5------------------------------------------
	bCQI <- bootclustrange(bfclara, biofam.seq, seqdist.args = list(method = "HAM"), R = 50, sample.size = 100,  parallel=TRUE)
	bCQI
  plot(bCQI, norm="zscore")

## ----seqdplotclust, fig.width=8, fig.height=8---------------------------------
seqdplot(biofam.seq, group=bfclara$clustering$cluster5)

## ----seqclarafuzzy, warning=FALSE, message=FALSE, fig.width=8, fig.height=5----
bfclaraf <- seqclararange(biofam.seq, R = 50, sample.size = 100, kvals = 2:10, method="fuzzy",
							seqdist.args = list(method = "HAM"), parallel=TRUE)
bfclaraf
plot(bfclaraf, norm="zscore")

## ----seqdplotclustf, dev="png", fig.width=8, fig.height=8---------------------
fuzzyseqplot(biofam.seq, group=bfclaraf$clustering$cluster4, type="I", sortv="membership", membership.threashold=0.4)

