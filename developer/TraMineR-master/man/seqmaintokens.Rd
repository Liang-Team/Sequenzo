\name{seqmaintokens}
\alias{seqmaintokens}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Indexes of most frequent tokens
%%  ~~function to do ... ~~
}
\description{Extracts the indexes of the most frequent token, either the \code{k} most frequent tokens or the tokens that occur on average more than \code{mint} times.
}
\usage{
seqmaintokens(seqdata, k=8L, mint=NULL, ...)
}
\arguments{
  \item{seqdata}{state sequence \code{stslist} object as produced by \code{\link{seqdef}}.
  }
  \item{k}{Integer. Number of main states.
  }
  \item{mint}{Real. Minimal mean number of occurrences per sequence.
  }
  \item{...}{Additional arguments passed to \code{\link{seqmeant}}
  }
}

\details{
When \code{mint} is \code{NULL}, indexes of the \code{k} most frequent tokens. Otherwise, indexes of tokens occurring on average more than \code{tmin} times are returned as long as their number does not exceed \code{k}. If more than \code{k}, indexes of the \code{k} most frequent are returned.
}

\value{
 Vector of indexes of the most frequent tokens repecting order of the alphabet.
}

\seealso{\code{\link{seqmeant}}
}

%%\references{
%% ~put references to the literature/web site here ~
%%}

\author{
Gilbert Ritschard
}

\examples{
data(biofam)
b.lab <- c("Parent",
           "Left",
           "Married",
           "Left+Marr",
           "Child",
           "Left+Child",
           "Left+Marr+Child",
           "Divorced"
)
b.short <- c("P","L","M","L+M","C","L+C","L+M+C","D")
set.seed(5)
cases <- sample(nrow(biofam),100)
b.seq <- seqdef(biofam[cases,10:25], labels=b.lab, states=b.short,
                weights=biofam[cases,"wp00tbgs"])

## Tokens occurring at least once on average
alphabet(b.seq)[seqmaintokens(b.seq, mint=1)]
#[1] "P"     "L"     "L+M"   "L+M+C"

## Three more frequent tokens
main.tokens <- seqmaintokens(b.seq, k=3)
## Labels of main tokens
attr(b.seq,"labels")[main.tokens]
#[1] "Parent"  "Left"    "Left+Marr+Child"
## Colors of main tokens
cpal(b.seq)[main.tokens]
#[1] "#7FC97F" "#BEAED4" "#BF5B17"} 