\name{seqindic.dyn}
\alias{seqindic.dyn}
\alias{print.dynin}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Dynamic index
%%  ~~function to do ... ~~
}
\description{Dynamic (i.e. successive) values of an individual index. For each sequence, the values of the selected index is computed on sliding windows.
}
\usage{
seqindic.dyn(seqdata, indic="cplx", window.size = .2, sliding=TRUE,
      wstep=1, with.missing=FALSE, endmiss.as.void=FALSE, silent.indic=TRUE, ...)
}
\arguments{
  \item{seqdata}{state sequence object (\code{stslist}) as produced by \code{seqdef}
  }
  \item{indic}{character string: the individual index. Can be any value supported by \code{\link[TraMineR]{seqindic}} except index group names.
  }
  \item{window.size}{integer or real. If an integer > 1, window size in number of positions. If real number in the range ]0,1), the window size is set as that proportion of the length of the longest sequence.
  }
  \item{sliding}{logical: Should indic be computed on sliding windows? If \code{FALSE}, windows are incremented starting with a window of size \code{window.size}.
  }
  \item{wstep}{integer: size of position gap between successive windows.
  }
  \item{with.missing}{logical. Should the missing state be treated as a state of the alphabet?
  }
  \item{endmiss.as.void}{logical. When \code{with.missing=FALSE}, should missings at end of windows be considered as voids, i.e. should the sequence end before end missings? When \code{FALSE} (default), the index is set as \code{NA} for windows that end with a void and, when \code{TRUE}, for windows that end with a void or a missing. Ignored when \code{with.missing=TRUE}.
  }
  \item{silent.indic}{logical. Should messages issued during computation of indic be suppressed?
  }
  \item{...}{additional arguments passed to \code{\link[TraMineR]{seqindic}}
  }
}

\details{
The function implements the dynamic sequence analysis approach of \cite{Pelletier et al. (2020)} and generalizes the method to any of the over 20 indicators provided by \code{\link[TraMineR]{seqindic}}.

The values of the \code{indic} index are computed for each sequence either on sliding windows of size \code{window.size} or on incremental windows starting from a first window of size \code{window.size}.

Column names refer to the end the windows.
}

\value{
A matrix of class \code{dynin} with attributes \code{xtstep}, \code{tick.last}, \code{weights}, \code{window.size}, \code{sliding}, and \code{indic}. The first three as well as the row and column names are taken from \code{seqdata}.

There are \code{print} and \code{plot} methods for \code{dynin} objects. See \code{\link{plot.dynin}}.
}

\seealso{\code{\link[TraMineR]{seqindic}}, \code{\link{plot.dynin}}
}

\references{
Pelletier, D., Bignami-Van Assche, S., & Simard-Gendron, A. (2020) Measuring Life Course Complexity with Dynamic Sequence Analysis, \emph{Social Indicators Research} \doi{10.1007/s11205-020-02464-y}.

%% ~put references to the literature/web site here ~
}

\author{
Gilbert Ritschard
}

\examples{
data(actcal)
cases <- 1:100
actcal <- actcal[cases,] ## Here, only a subset
actcal.seq <- seqdef(actcal[,13:24], alphabet=c('A','B','C','D'))

## Using windows every three positions
a.dyn <- seqindic.dyn(actcal.seq, indic='cplx', with.missing=FALSE, wstep=3)
plot(a.dyn, group=actcal[cases,'sex'])

## Trimmed mean (to illustrate fstat with specific arguments)
plot(a.dyn, group=actcal[cases,'sex'], fstat=function(x)mean(x, trim=.02))

## Incremental windows
ai.dyn <- seqindic.dyn(actcal.seq, indic='cplx', with.missing=FALSE, wstep=3,
                sliding=FALSE)
plot(ai.dyn, group=actcal[cases,'sex'])

#############
## Sequences of different lengths, and with missing values and weights
data(ex1)
s.ex1 <- seqdef(ex1[,1:13],weights=ex1[,"weights"])
seqlength(s.ex1)
seqlength(s.ex1, with.missing=FALSE)
group <- c(1,1,1,2,2,2,2)

ind.d <- seqindic.dyn(s.ex1, indic='cplx', with.missing=FALSE)
plot(ind.d, group=group, fstat=weighted.mean, na.rm=TRUE, conf=TRUE, ret=TRUE)

## Treating 'missing' as a regular state
ind.dm <- seqindic.dyn(s.ex1, indic='cplx', with.missing=TRUE)
plot(ind.dm, group=group, fstat=weighted.mean, na.rm=TRUE, conf=TRUE, ret=TRUE)
} 