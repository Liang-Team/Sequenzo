\name{seqconc}
\alias{seqconc}
\title{Concatenate vectors of states or events into a character string}
\description{
Concatenate vectors of states or events into a character string. In the string, each state is separated by 'sep'. The void elements in the input sequences are eliminated.
}
\usage{
seqconc(data, var=NULL, sep="-", vname="Sequence", void=NA)
}
\arguments{
  \item{data}{A data frame or matrix containing sequence data (tibble will be converted with \code{as.data.frame}).}

  \item{var}{List of the columns containing the sequences. Default is \code{NULL} in which case all  columns are retained. Whether the sequences are in the compressed (character strings) or extended format is automatically detected by counting the number of columns.}

  \item{sep}{Character used as separator. By default, "\code{-}".}

  \item{vname}{an optional name for the variable containing the sequences. By default, \code{"Sequence"}.}

	\item{void}{the code used for void elements appearing in the sequences (see \cite{Gabadinho et al. (2009)} for more details on missing values and void elements in sequences). Default is \code{NA}.}

}
\value{a vector of character strings, one for each row in the input data.}

\author{Alexis Gabadinho}

\references{
Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009). Mining Sequence Data in \code{R} with the \code{TraMineR} package: A user's guide. \emph{Department of Econometrics and Laboratory of Demography, University of Geneva}.
}

\seealso{
 \code{\link{seqdecomp}}.
}
\examples{
data(actcal)
actcal.string <- seqconc(actcal,13:24)
head(actcal.string)
}
\keyword{Data handling}
