\name{biofam}
\docType{data}
\alias{biofam}

\title{Example data set: Family life states from the Swiss Household Panel biographical survey}

\description{2000 16 year-long family life sequences built from the retrospective biographical survey carried out by the Swiss Household Panel (SHP) in 2002.}


\details{
The \emph{biofam} data set was constructed by Müller et al. (2007)
from the data of the retrospective biographical survey carried out by the Swiss Household Panel (SHP) in 2002.

The data set contains (in columns 10 to 25) sequences of family life states from age 15 to 30 (sequence length is 16) and a series of covariates. The sequences are a sample of 2000 sequences of those created from the SHP biographical survey. It includes only individuals who were at least 30 years old at the time of the survey.  The \emph{biofam} data set describes family life courses of 2000 individuals born between 1909 and 1972.

The states numbered from 0 to 7 are defined from the combination of five basic states, namely Living with parents (Parent), Left home (Left), Married (Marr), Having Children (Child), Divorced:\cr
\cr
0 = "Parent" \cr
1 = "Left" \cr
2 = "Married" \cr
3 = "Left+Marr" \cr
4 = "Child" \cr
5 = "Left+Child" \cr
6 = "Left+Marr+Child" \cr
7 = "Divorced" \cr

The covariates are: \cr

\tabular{ll}{
\code{sex} \tab \cr
\code{birthyr }  \tab (birth year)\cr
\code{nat_1_02}  \tab (first nationality) \cr
\code{plingu02}  \tab (language of questionnaire) \cr
\code{p02r01 }   \tab (religion)\cr
\code{p02r04 }   \tab (religious participation) \cr
\code{cspfaj }   \tab (father's social status) \cr
\code{cspmoj }   \tab (mother's social status) \cr
}

Two additional weights variables are inserted for illustrative purpose ONLY (since \code{biofam} is a subsample of the original data, these weights are not adapted to the actual data): \cr
\tabular{ll}{
\code{wp00tbgp} \tab (weights inflating to the Swiss population)\cr
\code{wp00tbgs} \tab (weights respecting sample size) \cr
}



}
\usage{data(biofam)

data(bfspell)
}
\format{A data frame with 2000 rows, 16 state variables, 1 id variable and 7 covariates and 2 weights variables.}
\source{Swiss Household Panel \url{https://forscenter.ch/projects/swiss-household-panel/}}
\references{
  Müller, N. S., M. Studer, G. Ritschard (2007).
  Classification de parcours de vie à l'aide de l'optimal matching.
  In \emph{XIVe Rencontre de la Société francophone de classification
  (SFC 2007), Paris, 5 - 7 septembre 2007}, pp. 157--160.
}
\keyword{Datasets}
