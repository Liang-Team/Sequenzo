\name{seqstart}
\alias{seqstart}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Aligning sequence data on a new start time.
}
\description{
Changing the position alignment of a set of sequences.
}
\usage{
seqstart(seqdata, data.start, new.start, tmin = NULL, tmax = NULL, missing = NA)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{a data frame or matrix containing sequence data.}
  \item{data.start}{Integer. The actual starting date of the sequences. In case of sequence-dependent start dates, should be a vector of length equal to the number of rows of seqdata.}
  \item{new.start}{Integer. The new starting date. In case of sequence-dependent start dates, should be a vector of length equal to the number of rows of seqdata.}
  \item{tmin}{Integer. Start position on new position axis. If \code{NULL}, it is guessed from the data.}
  \item{tmax}{Integer. End position on new position axis. If \code{NULL}, it is guessed from the data.}
  \item{missing}{Character. Code used to fill missing data in the new time axis.}
}
\value{A matrix.}


\author{Matthias Studer}

\note{Warning: This function needs further testing.}

\examples{

#An example data set
paneldata <- matrix(c("A" ,"A" , "B" , "B" , "B",
"A" , "A" , "B" , "B" , "B",
"A" , "A",  "B" , "B" , "B" ,
"A"  ,"A" , "A" , "B"  ,"B" ,
"A"  ,"A" , "A" , "A" , "B"), byrow=TRUE, ncol=5)
colnames(paneldata) <- 2000:2004

print(paneldata)

## Assuming data are aligned on calendar years, starting in 2000
## Change from calendar date to age alignment
startyear <- 2000
birthyear <- 1995:1999
agedata <- seqstart(paneldata, data.start=startyear, new.start=birthyear)
colnames(agedata) <- 1:ncol(agedata)
print(agedata)

## Retaining only ages between 3 and 7 (4th and 8th year after birthyear).
seqstart(paneldata, data.start=startyear, new.start=birthyear, tmin=4, tmax=8, missing="*")

## Changing back from age to calendar time alignment
ageatstart <- startyear - birthyear
seqstart(agedata, data.start=1,  new.start=ageatstart)
## Same but dropping right columns filled with NA's
seqstart(agedata, data.start=1,  new.start=ageatstart, tmax=5)

}
\keyword{data format}
