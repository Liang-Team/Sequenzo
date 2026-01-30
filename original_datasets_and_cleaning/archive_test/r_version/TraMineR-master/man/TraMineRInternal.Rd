\name{TraMineRInternal}
%%\alias{TraMineRInternalCheckArgs}
\alias{TraMineRInternalLayout}
\alias{TraMineRInternalLegend}
\alias{TraMineRInternalNodeInit}
\alias{TraMineRInternalSeqeage}
\alias{TraMineRInternalSeqgbar}
\alias{TraMineRInternalSplitInit}
\alias{TraMineRInternalWeightedInertiaDist}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Access to TraMineR internal functions}
\description{
  Functions allowing other packages to access some TraMineR internal functions.
  Corresponding functions are respectively \code{TraMineR.setlayout},
  \code{TraMineR.Legend}, \code{DTNInit}, \code{seqeage}, \code{seqgbar},
  \code{DTNsplit}, and \code{tmrWeightedInertiaDist}. For experts only.
}
\usage{
%%TraMineRInternalCheckArgs(...)
TraMineRInternalLayout(...)
TraMineRInternalLegend(...)
TraMineRInternalNodeInit(...)
TraMineRInternalSeqeage(...)
TraMineRInternalSeqgbar(...)
TraMineRInternalSplitInit(...)
TraMineRInternalWeightedInertiaDist(diss, diss.size, is.dist, individuals, sweights, var)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{Arguments passed to or from other methods.}
  \item{diss}{See tmrWeightedInertiaDist().}
  \item{diss.size}{See tmrWeightedInertiaDist().}
  \item{is.dist}{See tmrWeightedInertiaDist().}
  \item{individuals}{See tmrWeightedInertiaDist().}
  \item{sweights}{See tmrWeightedInertiaDist().}
  \item{var}{See tmrWeightedInertiaDist().}
}
