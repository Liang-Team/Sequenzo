\name{seqpolyads}
\alias{seqpolyads}

\title{Measuring the Degree of Within-Polyadic Similarities}

\description{The function computes measures of the degree of similarities within polyadic member sequences compared to randomly assigned polyadic member sequences.}
\usage{
seqpolyads(seqlist, a=1, method="HAM", ...,
    w=rep(1,ncol(combn(1:length(seqlist),2))),
    s=36963, T=1000, core=1, replace=TRUE, weighted=TRUE,
    with.missing=FALSE, rand.weight.type=1, role.weights=NULL,
    show.time=FALSE)
}
\arguments{
\item{seqlist}{A list of \var{J>1} state sequence \code{stslist} objects. List of input sets (polyads) of polyadic sequences. The state sequence objects in the list must all have the same number \var{N} of sequences and the same alphabet.  The state sequence objects should be created with \code{\link[TraMineR]{seqdef}} and the list with \code{\link{list}}. E.g., \code{list(gen1.seq,}\code{gen2.seq,}\code{gen3.seq)}.
}

\item{a}{Integer, 1 or 2. Random generation mechanism. If 1 (default), draws from the observed set of sequences, and if 2, in addition random draws of states from each randomly drawn sequence. See reference below for detail.
}
\item{method}{String. Method for computing sequence distances. See \code{\link[TraMineR]{seqdist}}. Additional arguments may be required depending on the method chosen.
}

\item{...}{Additional arguments passed to \code{\link[TraMineR]{seqdist}}
}

\item{s}{Integer. Default 36963. Using the same seed number on the same computer guarantees the same results each time. Set \code{s=NULL} if you don't want to set a seed. The random generator can be chosen with \code{\link{RNGkind}}.
}

\item{w}{Integer vector. Default 1. The weights assigned to between-polyadic member sets in the weight matrix. For example, for dyadic sequences, no weight is necessary and the distance computation takes on the default of 1. For triadic sequences, there are three weights between the first and the second members, the first and the third members, and the second and the third members, in a row-wise order. See reference below.
}

\item{T}{Integer. Default 1,000. The number of randomized computations.
}

\item{core}{Integer. Default 1. Number of cores for the computation. When greater than 1, the procedure utilizes parallel processing.
}
  \item{replace}{Logical. When \code{a=2}, should state sampling in each sequence be done with replacement? Default is \code{TRUE}. Ignored when \code{a=1}.
  }
  \item{weighted}{Logical. Should we account for the weights when present in the sequence objects? See details. Default is \code{TRUE}.
  }
  \item{with.missing}{Logical. Should the missing state be considered as a regular state? Default is \code{FALSE}.
  }
  \item{rand.weight.type}{Integer, 1 or 2. Ignored when \code{weighted=FALSE}. If 1 (default), weight of each randomized polyad is the average of original weights of its members. If 2, member weights are adjusted by dividing them by the sum of weights of all drawn members of the same type.
  }
  \item{role.weights}{\code{NULL} or vector of non-negative weights of same length as the list \code{seqlist}. Ignored when \code{weighted=FALSE}. If non null, role weights for determining the weights of the randomized polyads.
      }
  \item{show.time}{Logical. Should elapsed time be displayed? Default is \code{FALSE}.
  }
}

\details{
The function computes the polyadic distance of the observed polyads, i.e., the (weighted) mean of the pairwise distances between members of the polyad. In addition, the following statistics are computed:

The \var{U} statistic measures for each observed polyad by how much its polyadic distance differs from the mean polyadic distance of \code{T} randomized polyads. \var{U.tp} is the \var{p}-value for a two-tailed \var{t}-test of the \var{U} statistic.

The \var{V} statistic is, for each observed polyad, the proportion of \code{T} randomized polyads that have a greater polyadic distance. \var{V.95} is an associated dummy that takes value 1 when the proportion \var{V} is greater than 95\% and 0 otherwise.

When the sequence objects in \code{seqlist} have weights and \code{weighted=TRUE}, the randomized sequences are sampled using the weights of the first element in the list. Each member of an observed polyad is supposed to have the same weight. This does not hold for the randomized polyads that are obtained by sampling their members independently. The weights of each randomized sequence is set as the average of the weights of its members.  When role weights are provided with \code{role.weights}, a weighted average of the member weights is used. When \code{rand.weight.type=1}, original member weights are used. When \code{rand.weight.type=2}, the weights of randomly selected members are adjusted by the sum of weights of all randomly drawn members of the same type.

When \code{core > 1}, the function uses the \code{doParallel} package for parallel computation.
}

\value{The function outputs a list of seven objects:

\item{mean.dist}{Vector of length 2 with the average observed and random within-polyadic distances.
}


\item{U}{Vector of \var{N} number of \var{U} statistics (see reference).
}

\item{U.tp}{Vector of \var{N} number of \var{p}-values for a two-tailed \var{t}-test of the \var{U} statistic.
}

\item{V}{Vector of \var{N} number of \var{V} statistics (see reference).
}

\item{V.95}{Vector of \var{N} number of 1s or 0s: 1 if a \var{V} value is at least 95 percent confident, 0 otherwise.
}

\item{observed.dist}{Vector of within-polyadic distances for the observed  polyadic members.
}

\item{random.dist}{Vector of within-polyadic distances for the \var{T} number of randomly matched polyadic members.
}
}

\examples{
data(polyads)
Gen <- polyads$Gen
seqGrandP <- seqdef(polyads[Gen=="1st Generation",2:11])
seqParent <- seqdef(polyads[Gen=="2nd Generation",2:11])
seqChild <- seqdef(polyads[Gen=="3rd Generation",2:11])
Seq <- rbind(seqGrandP,seqParent,seqChild)
slgth <- ncol(Seq)
colnames(Seq) <- 21:30
seqIplot(Seq,group=Gen,idxs=10:1,ylab="Triad",xlab="Age")
seqL <- list(seqGrandP,seqParent,seqChild)
core=1
seqG2.Tim <- seqpolyads(seqL[1:2],method="HAM",a=1,core=core,T=100)
seqG3.Tim <- seqpolyads(seqL,method="HAM",a=1,core=core,T=100)
seqG2.Dur <- seqpolyads(seqL[1:2],method="CHI2",step=slgth,core=core,T=100)
seqG3.Dur <- seqpolyads(seqL,method="CHI2",step=slgth,core=core,T=100)

}

\references{
Tim F. Liao (2021), "Using Sequence Analysis to Quantify How Strongly Life Courses Are Linked.” \emph{Sociological Science} \bold{8}, 48-72, \doi{10.15195/v8.a3}.
}
\author{Tim Liao and Gilbert Ritschard} 