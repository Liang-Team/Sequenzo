\name{seqrecode}
\alias{seqrecode}
\alias{recodef}

\title{
Recoding state sequence objects and factors
}
\description{
	Utilities for recoding factors or state sequence objects created with \code{\link{seqdef}}.
}
\usage{
seqrecode(seqdata, recodes, otherwise = NULL,
     labels = NULL, cpal = NULL)
recodef(x, recodes, otherwise=NULL, na=NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{
	The state sequence object to be recoded (created with \code{\link{seqdef}}).
	}
  \item{recodes}{A \code{list} specifying the recoding operations where each element is in the form \code{newcode=oldcode} or \code{newcode=c(oldcode1, oldcode2,...)}. The rules are treated in the same order as they appear, hence subsequent rules may modify the first ones.}
  \item{otherwise}{\code{NULL} or Character. Level given to cases uncovered by the \code{recodes} list. If \code{NULL}, old states remain unchanged.}
  \item{labels}{optional state labels used for the color legend of TraMineR's graphics. If \code{NULL} (default), the state names in the alphabet are also used as state labels (see \code{\link{seqdef}}).}
  \item{cpal}{an optional color palette for representing the newly defined alphabet in graphics. If \code{NULL} (default), a color palette is created from the colors in \code{seqdata} by assigning to \code{newcode} the color of the first old state listed as \code{oldcode} and by leaving the colors of the other states unchanged.
  }
  \item{x}{A factor to be recoded.}
  \item{na}{Character vector. If not \code{NULL}, the list of states that should be recoded as \code{NA} (missing values).}
}

%\details{}

\value{ The recoded factor or state sequence object.}

\seealso{
\code{\link{seqdef}} to create a state sequence object.
}

\examples{
## Recoding a state sequence object with seqrecode
data(actcal)
## Creating a state sequence object
actcal.seq <- seqdef(actcal,13:24, labels=c("> 37 hours", "19-36 hours",
   "1-18 hours", "no work"))
## Regrouping states B and C and setting the whole alphabet to A BC D
actcal.new <-seqrecode(actcal.seq,
    recodes = list("A"="A", "BC"=c("B", "C"), "D"="D"))
## Crosstabulate the first column of the recoded and
## original state sequence objects
table(actcal.new[,1], actcal.seq[,1])

## Same as before but using automatically original
## codes for unspecified states.
actcal.new2 <-seqrecode(actcal.seq,
	recodes = list("BC"=c("B", "C")))
table(actcal.new2[,1], actcal.seq[,1])

## Same as before but using otherwise
actcal.new3 <-seqrecode(actcal.seq, recodes = list("A"="A", "D"="D"),
   otherwise="BC")
table(actcal.new3[,1], actcal.seq[,1])

## Recoding factors
## Recoding the marital status to oppose married to all other case
maritalstatus <- recodef(actcal$civsta00,
	recodes=list("Married"="married"), otherwise="Single")
summary(maritalstatus)
table(maritalstatus, actcal$civsta00)

## Recoding the number of kids in the household
## -2 is a missing value
nbkids <- recodef(actcal$nbkid00,
   recodes=list("None"=0, "One"=1, "Two or more"=2:10), na=-2)
table(nbkids, actcal$nbkid00, useNA="always")

}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Data handling}
