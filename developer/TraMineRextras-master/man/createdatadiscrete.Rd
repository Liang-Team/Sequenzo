\encoding{latin1}
\name{createdatadiscrete}
\alias{createdatadiscrete}
\title{Transform time to event data into a discrete data format}

\description{
Transform time to event data (in a specific format, see the details below) into a person-period data format suitable for automatic sequential association rules extraction
 }

\usage{
createdatadiscrete(ids, data, vars, agemin, agemax,
    supvar=NULL)
}

\arguments{
  \item{ids}{a vector containing an unique identification number for each case}
  \item{data}{a data frame containing time to event data, with variables containing the durations named as in the vars argument, and those with the censoring indicators named as in the vars argument followed by "ST" (for example column A is duration until event A, and column AST is the censoring indicator). This data frame must contain an unique identification variable named "IDPERS".}
  \item{vars}{a vector with the names of the duration variables}
  \item{agemin}{a data frame with two variables : "IDPERS" for the unique identification variable, and "AGE" for the starting time of the observation}
  \item{agemax}{a data frame with two variables : "IDPERS" for the unique identification variable, and "AGE" for the ending time of the observation}
  \item{supvar}{a vector of variables to add to the resulting person-period data frame}
}

\details{The data frame from the \code{data} argument must contain two variables for each event: a duration variable that indicates the time when the event occurred, and a status variable that indicates if the event occurred (1) or not (0). If the event did not occur, the observation for this individual will go until the age specified through the \code{agemax} argument. Each status variable must have the name of the corresponding duration variable suffixed by "ST". For example, if the duration variable for an event "divorce" is called "div", then the status variable has to be named "divST". \cr

The result from this function is a list with one person-period data frame by event, where the dependent event is different each time. Please see the attached data file and code for an example.

The resulting object is one of the required argument for the \code{seqerulesdisc} function that computes the association rules, the hazard ratios and the p-values, using discrete-time regressions. Unlike the method presented in Müller et al. 2010, this function does not use Cox proportional hazard models, but discrete-time regression models with a complementary log-log link function, which gives similar results.
 }

 \author{Nicolas S. Müller}

\seealso{
 \code{\link{seqerulesdisc}} to compute the association rules.}

\value{a list with one person-period data frame by event, where the dependent event is different each time. Please see the attached data file and code for an example.}

\references{
Müller, N.S., M. Studer, G. Ritschard et A. Gabadinho (2010), Extraction de règles d'association séquentielle à l'aide de modèles semi-paramétriques à risques proportionnels, \emph{Revue des Nouvelles Technologies de l'Information}, \bold{Vol. E-19}, EGC 2010, pp. 25-36
 }

\examples{
##
}
