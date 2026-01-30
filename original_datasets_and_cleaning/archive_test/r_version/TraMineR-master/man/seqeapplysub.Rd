\name{seqeapplysub}
\alias{seqeapplysub}
\title{Checking for the presence of given event subsequences}
\description{
Checks occurrences of the subsequences \code{subseq} among the event sequences and returns the result according to the selected \code{method}.
}
\usage{
seqeapplysub(subseq, method = NULL, constraint = NULL,
            rules=FALSE)
}
\arguments{
 \item{subseq}{list of subsequences (an event subsequence object) such as created by \code{\link{seqefsub}}}
 \item{method}{type of result, should be one of \code{"count"}, \code{"presence"} or \code{"age"}}
 \item{constraint}{
	Time constraints overriding those used to compute \code{subseq}. See \code{\link{seqeconstraint}}
	}
 \item{rules}{If set to \code{TRUE}, instead of checking occurrences of the subsequences among the event sequences, check the occurrence of the subsequences inside the subsequences (internally used by \code{seqerules})}
}
\details{
  There are three methods implemented:
  \code{"count"} counts the number of occurrence of each given subsequence in each event sequence;
  \code{"presence"} returns 1 if the subsequence is present, 0 otherwise;
  \code{"age"} returns the age of appearance of each subsequence in each event sequence.
In case of multiple possibilities, the age of the first occurrence is returned. When the subsequence is not in the sequence, -1 is returned.
}

\value{
The return value is a matrix where each row corresponds to a sequence (row names are set accordingly) and each column corresponds to a subsequence (col names are set accordingly).
The cells of the matrix contain the requested values (count, presence-absence indicator or age).
}

\seealso{\code{\link{seqecreate}} for more information on event sequence object and \cite{Gabadinho et al. (2009)} on how to use the event sequence analysis module.}

\references{
Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009). Mining Sequence Data in \code{R} with the \code{TraMineR} package: A user's guide. Department of Econometrics and Laboratory of Demography, University of Geneva.
}


\examples{
## Loading data
data(actcal.tse)

## Creating the event sequence object
actcal.eseq <- seqecreate(actcal.tse)

## Printing sequences
actcal.eseq[1:10]

## Looking for frequent subsequences
fsubseq <- seqefsub(actcal.eseq,pmin.support=0.01)

## Counting the number of occurrences of each subsequence
msubcount <- seqeapplysub(fsubseq,method="count")
## First lines...
msubcount[1:10,1:10]
## Presence-absence of each subsequence
msubpres <- seqeapplysub(fsubseq,method="presence")
## First lines...
msubpres[1:10,1:10]

## Age at first appearance of each subsequence
msubage <- seqeapplysub(fsubseq,method="age")

## First lines...
msubage[1:10,1:10]
}
\author{Matthias Studer and Reto Bürgin (alternative counting methods) (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
