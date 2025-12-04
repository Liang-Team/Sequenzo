\name{mvad}
\docType{data}
\alias{mvad}
\title{Example data set: Transition from school to work}

\description{ The data comes from a study by McVicar and Anyadike-Danes on
transition from school to work. The data consist of static background
characteristics and a time series sequence of 72 monthly labour market
activities for each of 712 individuals in a cohort survey. The individuals
were followed up from July 1993 to June 1999. The monthly states are recorded
in columns 15 (\code{Jul.93})  to 86 (\code{Jun.99}).

States are:\cr

\tabular{ll}{
\code{employment}  \tab (EM) \cr
\code{FE}          \tab further education (FE)\cr
\code{HE}          \tab higher education (HE) \cr
\code{joblessness} \tab (JL) \cr
\code{school}      \tab (SC) \cr
\code{training}    \tab (TR)\cr
}

The data set contains also ids (\code{id}) and sample weights (\code{weight}) as well as the following binary covariates:\cr
\cr
\code{male}\cr
\code{catholic}\cr
\code{Belfast}, \code{N.Eastern}, \code{Southern}, \code{S.Eastern}, \code{Western} (location of school, one of five Education and Library Board areas in Northern Ireland)\cr
\code{Grammar} (type of secondary education, 1=grammar school)\cr
\code{funemp} (father's employment status at time of survey, 1=father unemployed)\cr
\code{gcse5eq} (qualifications gained by the end of compulsory education, 1=5+ GCSEs at grades A-C, or equivalent)\cr
\code{fmpr} (SOC code of father's current or most recent job, 1=SOC1 (professional, managerial or related))\cr
\code{livboth} (living arrangements at time of first sweep of survey (June 1995), 1=living with both parents)\cr
}

\usage{data(mvad)}
\format{A data frame containing 712 rows, 72 state variables, 1 id variable and 13 covariates.}
\source{McVicar and Anyadike-Danes (2002)}
\references{
McVicar, Duncan and Anyadike-Danes, Michael (2002). Predicting Successful and Unsuccessful Transitions from School to Work by Using Sequence Methods,
\emph{Journal of the Royal Statistical Society. Series A (Statistics in Society)}, 165, 2, pp. 317--334.
}
\keyword{Datasets}
