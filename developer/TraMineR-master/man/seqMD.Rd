\name{seqMD}
\alias{seqMD}
\alias{seqdistmc}
\title{Multidomain sequences}
\description{Build multidomain (MD) sequences of combined individual domain states (expanded alphabet), derive multidomain indel and substitution costs from domain costs by means of an additive trick (CAT), and compute OM pairwise distances using CAT costs.
}

\usage{
seqMD(channels,
  method=NULL,
  norm="none",
  indel="auto",
  sm=NULL,
  with.missing=NULL,
  full.matrix=TRUE,
  link="sum",
  cval=2,
  miss.cost=2,
  cweight=NULL,
  what="MDseq",
  ch.sep="+",
  fill.with.miss=TRUE
  )

seqdistmc(channels, what="diss", ch.sep="@@@@TraMineRSep@@@@", ...)

}


\arguments{
 \item{channels}{A list of domain state sequence \code{stslist} objects defined
    with the \code{\link{seqdef}} function, each state sequence object corresponding
    to a domain.}

 \item{method}{String. Default: \code{NULL}. Dissimilarity measure between sequences.
    When \code{what="diss"}, must be one of \code{"OM"} (Optimal Matching),
    \code{"HAM"} (Hamming distance), or \code{"DHD"} (Dynamic Hamming distance). Otherwise, ignored. Can also be \code{"LCS"} (Longest common subsequence), but see details. }

 \item{norm}{
    String.
    Default: \code{"none"}.
    The normalization method to use. See \code{\link{seqdist}}. Ignored if \code{what} is not \code{"diss"}.
    }

 \item{indel}{Double, vector of doubles, or list with an insertion/deletion cost or a vector of state dependent indel costs for each domain. Can also be \code{"auto"} (default), in which case the indel cost of each domain is automatically set in accordance with the \code{sm} value of the domain. See \code{indel} argument of \code{\link{seqdist}}.
     }

 \item{sm}{A list with a substitution-cost matrix for each domain
   or a list of method names for generating the domain substitution costs
   (see \code{\link{seqcost}}). Ignored when \code{method="LCS"}.
   }

 \item{with.missing}{Logical, vector of logical, or \code{NULL} (default). See \code{\link{seqdist}} and \code{\link{seqcost}}.
   }

 \item{full.matrix}{Logical. If \code{TRUE} (default), the full distance matrix between MD sequences is returned.
   If \code{FALSE}, an object of class \code{\link{dist}} is returned.
   }

 \item{link}{Character string. One of \code{"sum"} or \code{"mean"}. Method to compute the "link"
   between domains. Default is to sum substitution and indel costs.
   }

 \item{cval}{Double. Domain substitution cost for \code{"CONSTANT"} matrix, see \code{\link{seqcost}}.}

 \item{miss.cost}{Double. Cost to substitute missing values at domain level, see \code{\link{seqcost}}.}

 \item{cweight}{A vector of domain weights. Default is 1 (same weight for each domain).}

 \item{what}{Character string. What output should be returned? One of \code{"MDseq", "cost", "diss"}. The deprecated value \code{what="sm"} is treated as \code{what="cost"}. \code{MDseq} returns the multidomain sequences expressed in terms of the expanded alphabet, \code{"cost"} the CAT costs, and "diss" the CAT-based multidomain distances.}

 \item{ch.sep}{Character string. Separator used for building state names of the expanded alphabet.}

 \item{fill.with.miss}{Logical. Should shorter domain sequences be filled with missings to match sequence lengths across domains? Applies only to domains that already have missings.}

 \item{...}{arguments passed to \code{seqMD}}
}

\details{The \code{seqMD} function builds MD sequences by combining the domain states. When \code{what="cost"}, it derives multidomain indel and substitution costs from the indel and substitution costs of each domain by means of the cost additive trick (CAT) (\cite{Ritschard et al., 2023}, \cite{Pollock, 2007}). When \code{what="diss"}, it computes multidomain distances using the CAT multidomain costs. The available metrics (see \code{method} argument) are optimal matching (\code{"OM"}), Hamming distance (\code{"HAM"}), and Dynamic Hamming Distance (\code{"DHD"}). If \code{method="LCS"}, distances are obtained with OM using CAT costs derived from domain indel and sm costs of respectively 1 and 2 (i.e. inputted \code{indel} and \code{sm} are ignored).  For other edit distances, extract the combined state sequence object (by setting \code{what="MDseq"}) and the CAT-multidomain substitution and indel costs (by setting \code{what="cost"}). Then use these outcomes as input in a call to \code{seqdist}.  See \code{\link{seqdist}} for more information about available distance measures.

Normalization may be useful when dealing with sequences that are not all of the same length. For details on the applied normalization, see \code{\link{seqdist}}.

Sequences lengths are supposed to match across domains. If \code{fill.with.miss} is \code{TRUE} and the i-th sequence is shorter in one domain than the longest i-th sequence, it will, when constructing the i-th MD sequence, be filled with missing values to adapt its length to that of the longest sequence. However, this applies only for domain that already have missings, i.e., domains with a corresponding \code{with.missing} value set as \code{TRUE}. }

\value{
When \code{what="MDseq"}, the MD sequences of combined states as a \code{stslist} sequence object.\cr
When \code{what="cost"}, the matrix of CAT-substitution costs with three attributes: \code{indel} the CAT-indel cost(s), \code{alphabet} the alphabet of the combined state sequences, and \code{cweight} the channel weights used.\cr
When \code{what="diss"}, a matrix of pairwise distances between MD sequences.
}

\seealso{ \code{\link{seqcost}}, \code{\link{seqdef}}, \code{\link{seqdist}}, \code{\link{seqplotMD}}. }

\references{
Ritschard, G., T.F. Liao, and E. Struffolino (2023). Strategies for
multidomain sequence analysis in social research.
\emph{Sociological Methodology}, 53(2), 288-322. \doi{10.1177/00811750231163833}.

 Pollock, G. (2007) Holistic trajectories: a study of combined employment, housing and family careers by using multiple-sequence analysis. \emph{Journal of the Royal Statistical Society: Series A} \bold{170}, Part 1, 167--183.
}

\examples{
data(biofam)

## Building one channel per type of event left home, married, and child
cases <- 200
bf <- as.matrix(biofam[1:cases, 10:25])
left <- bf==1 | bf==3 | bf==5 | bf==6
married <- bf == 2 | bf== 3 | bf==6
children <-  bf==4 | bf==5 | bf==6

## Building sequence objects
left.seq <- seqdef(left)
marr.seq <- seqdef(married)
child.seq <- seqdef(children)
channels <- list(LeftHome=left.seq, Marr=marr.seq, Child=child.seq)

## CAT multidomain distances based on channel specific cost methods
MDdist <- seqMD(channels, method="OM",
    sm =list("INDELSLOG", "INDELSLOG", "TRATE"), what="diss")

## Providing channel specific substitution costs
smatrix <- list()
smatrix[[1]] <- seqsubm(left.seq, method="TRATE")
smatrix[[2]] <- seqsubm(marr.seq, method="CONSTANT")
smatrix[[3]] <- seqsubm(child.seq, method="CONSTANT")

## Retrieving the MD sequences
MDseq <- seqMD(channels)
alphabet(MDseq)

## Retrieving the CAT multidomain substitution costs
## Using double weight for domain "child"
CATcost <- seqMD(channels,
    sm=smatrix, cweight=c(1,1,2), what="cost")

## OMspell distances between MD sequences
MDdist2 <- seqdist(MDseq, method="OMspell",
    sm = CATcost, indel=attr(CATcost,"indel"))

}


\author{Gilbert Ritschard and Matthias Studer}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity measures}
