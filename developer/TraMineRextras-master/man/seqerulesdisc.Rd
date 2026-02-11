\encoding{latin1}
\name{seqerulesdisc}
\alias{seqerulesdisc}
\title{Extract association rules using discrete time regression models}

\description{
Extract association rules from an object created by the \code{createdatadiscrete} function, using discrete time regression models to assess the significance of the extracted rules.
 }

\usage{
seqerulesdisc(fsubseq, datadiscr, tsef, pvalue=0.1, supvars=NULL, 
   adjust=TRUE, topt=FALSE, link="cloglog", dep=NULL)
}

\arguments{
  \item{fsubseq}{an object created using the \code{\link[TraMineR]{seqefsub}} function and that contains the list of subsequences to be tested for an association}
  \item{datadiscr}{the object created by the \code{\link{createdatadiscrete}} function and that contains the person-period data}
  \item{tsef}{the data frame containing the original time-to-event dataset (equivalent to the \code{data} argument of the \code{createdatadiscrete} function)}
  \item{pvalue}{the default threshold p-value to consider an association rule as significative, default is 0.1}
  \item{supvars}{a vector of variable names to be used as control variables in the regression models (experimental)}
  \item{adjust}{if set to TRUE, a Bonferroni adjustment is applied to the p-value threshold specified in the \code{pvalue} argument}
  \item{topt}{if set to TRUE, use an alternative algorithm to extract the rules (very experimental) ; default to FALSE}
  \item{link}{the link function to be used in the generalized linear regression model. To obtain hazard ratios, use the complementary log-log link function ("cloglog", as default). The other choice is to use a logit link function ("logit").}
  \item{dep}{if set to NULL, test all possible association rules. If an event is set, the function will only test association rules ending with this event}
}

\details{This function uses a list of subsequences created by the \code{seqefsub} function from the TraMineR package and tests each possible association rules. It then shows the association rules whose significance, assessed using a discrete time regression model, is higher than the specified p-value threshold.

The algorithm is described in the Müller et al. (2010) article, even though this function uses a discrete time regression model instead of the Cox regression model described in the article. A more complete explanation of the method is available in Müller (2011).
 }

 \author{Nicolas S. Müller}

\seealso{
 \code{\link{createdatadiscrete}} to create the object needed as the \code{datadiscr} argument.
 \code{\link[TraMineR]{seqefsub}} to create the object needed as the \code{fsubseq} argument.
 }

\value{a list with one person-period data frame by event, where the dependent event is different each time. Please see the attached data file and code for an example.}

\references{
Müller, N.S., M. Studer, G. Ritschard et A. Gabadinho (2010), Extraction de règles d'association séquentielle à l'aide de modèles semi-paramétriques à risques proportionnels, \emph{Revue des Nouvelles Technologies de l'Information}, \bold{Vol. E-19}, EGC 2010, pp. 25-36.

Müller, N.S. (2011), Inégalités sociales et effets cumulés au cours de la vie : concepts et méthodes, \emph{Thèse de doctorat, Faculté des sciences économiques et sociales, Université de Genève}, http://archive-ouverte.unige.ch/unige:17746.
 }

\examples{
##
}
