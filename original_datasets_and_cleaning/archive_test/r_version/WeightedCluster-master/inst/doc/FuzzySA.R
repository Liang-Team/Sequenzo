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

## ----message=FALSE------------------------------------------------------------
diss <- seqdist(biofam.seq, method="LCS")

## ----fannyclust, warning=FALSE, message=FALSE---------------------------------
library(cluster) ## Loading the library
fclust <- fanny(diss, k=5, diss=TRUE, memb.exp=1.5)

## -----------------------------------------------------------------------------
summary(fclust$membership)

## ----plotfd, fig.width=8, fig.height=5----------------------------------------
## Displaying the resulting clustering with membership threshold of 0.4
fuzzyseqplot(biofam.seq, group=fclust$membership, type="d")

## ----plotf, fig.width=8, fig.height=5-----------------------------------------
## Displaying the resulting clustering with membership threshold of 0.4
fuzzyseqplot(biofam.seq, group=fclust$membership, type="I", membership.threashold =0.4, sortv="membership")

## ----dreg---------------------------------------------------------------------
library(DirichletReg)
##Estimation of Dirichlet Regression
##Dependent variable formatting
fmember <- DR_data(fclust$membership)
## Estimation
bdirig <- DirichReg(fmember~sex+birthyr|1, data=biofam, model="alternative")
## Displaying results of Dirichlet regression.
summary(bdirig)

## ----betareg------------------------------------------------------------------
library(betareg)
## Estimation of beta regression
breg1 <- betareg(fclust$membership[, 3]~sex+birthyr, data=biofam)
## Displaying results
summary(breg1)

## -----------------------------------------------------------------------------
pclust <- seqpropclust(biofam.seq, diss=diss, maxcluster=5, properties=c("state", "duration"))
pclust

## ----eval=FALSE---------------------------------------------------------------
# seqtreedisplay(pclust, type="d", border=NA, showdepth=TRUE)

## -----------------------------------------------------------------------------
pclustqual <- as.clustrange(pclust, diss=diss, ncluster=5)
pclustqual

## ----fig.width=8, fig.height=5------------------------------------------------
seqdplot(biofam.seq, pclustqual$clustering$cluster4)

## ----include=FALSE------------------------------------------------------------
knitr::write_bib(file = 'packages.bib')

