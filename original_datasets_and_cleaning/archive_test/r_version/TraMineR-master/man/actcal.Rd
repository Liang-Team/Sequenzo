\name{actcal}
\docType{data}
\alias{actcal}
\title{Example data set: Activity calendar from the Swiss Household Panel}

\description{
This data set contains 2000 individual sequences of monthly activity statuses from
January to December 2000.}

\details{
The data set is a subsample of the data collected by the Swiss Household Panel (SHP).

The state column (variable) names are `\code{jan00}', `\code{feb00}', etc... and correspond to columns 13 to 24.

There are four possible states:\cr
\cr
A = Full-time paid job (> 37 hours)\cr
B = Long part-time paid job (19-36 hours) \cr
C = Short part-time paid job (1-18 hours) \cr
D = Unemployed (no work) \cr

The data set contains also the following covariates:\cr

\tabular{ll}{
\code{age00}    \tab (age in 2000) \cr
\code{educat00} \tab (education level) \cr
\code{civsta00} \tab (civil status) \cr
\code{nbadul00} \tab (number of adults in household) \cr
\code{nbkid00}  \tab (number of children) \cr
\code{aoldki00} \tab (age of oldest kid) \cr
\code{ayouki00} \tab (age of youngest kid) \cr
\code{region00} \tab (residence region) \cr
\code{com2.00}  \tab (residence commune type) \cr
\code{sex}      \tab (sex of respondent)\cr
\code{birthy}   \tab (birth year) \cr
}
}
\usage{data(actcal)}
\format{A data frame with 2000 rows, 12 state variables, 1 id variable and 11 covariates.}
\source{Swiss Household Panel}
\references{\url{https://forscenter.ch/projects/swiss-household-panel/}}
\keyword{Datasets}
