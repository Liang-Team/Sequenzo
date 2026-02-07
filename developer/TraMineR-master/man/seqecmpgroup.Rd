\name{seqecmpgroup}
\alias{seqecmpgroup}
\title{Identifying discriminating subsequences}
\description{
  Identify and sort the most discriminating subsequences by their discriminating power.
}
\usage{
seqecmpgroup(subseq, group, method="chisq", pvalue.limit=NULL,
             weighted = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{subseq}{A \code{subseqelist} object (list of subsequences) such as produced by \code{\link{seqefsub}}}
  \item{group}{Group membership, i.e., a variable or factor defining the groups which we want to discriminate}
  \item{method}{The discrimination method; one of \code{"bonferroni"} or \code{"chisq"}}
  \item{pvalue.limit}{Can be used to filter the results. Only subsequences with a p-value lower than this parameter are selected. If \code{NULL} all subsequences are returned (regardless of their p-values).}
  \item{weighted}{Logical. If \code{TRUE}, \code{seqecmpgroup} uses the weights specified in \code{subseq}, (see \code{\link{seqefsub}}).}
}
\details{
The following discrimination test functions are implemented:
  \code{chisq}, the Pearson Independence Chi-squared test, and
  \code{bonferroni}, the Pearson Independence Chi-squared test with Bonferroni correction.
}
\value{
  An objet of type \code{subseqelistchisq} (subtype of \code{subseqelist}) with the following elements
  \item{subseq}{Sorted list of found discriminating subsequences}
  \item{eseq}{The event sequence object on which the tests were computed}
  \item{constraint}{Time constraints used for searching the subsequences (see \code{\link{seqeconstraint}})}
  \item{labels}{Levels (value labels) of the target group variable}
  \item{type}{Type of test used}
  \item{data}{A data frame with columns support, index (original rank of the subsequence, i.e., its position in the inputted \code{subseq}) and a pair of frequency and Pearson residual columns for each group}
}

\references{
    Studer, M., Müller, N.S., Ritschard, G. & Gabadinho, A. (2010), "Classer, discriminer et visualiser des séquences d'événements", In Extraction et gestion des connaissances (EGC 2010), \emph{Revue des nouvelles technologies de l'information} RNTI. Vol. E-19, pp. 37-48.

    Ritschard, G., Bürgin, R., and Studer, M. (2014), "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) \emph{Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences}. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.

}

%\author{ ~~who you are~~ }
%\note{ ~~further notes~~

% ~Make other sections like Warning with \section{Warning }{....} ~
%}
\seealso{See also \code{\link{plot.subseqelistchisq}} to plot the results}
\examples{
data(actcal.tse)
actcal.eseq <- seqecreate(actcal.tse)

##Searching for frequent subsequences, that is, appearing at least 20 times
fsubseq <- seqefsub(actcal.eseq, pmin.support=0.01)

##searching for susbsequences discriminating the most men and women
data(actcal)
discr <- seqecmpgroup(fsubseq, group=actcal$sex, method="bonferroni")
##Printing the six most discriminating subsequences
print(discr[1:6])
##Plotting the six most discriminating subsequences
plot(discr[1:6])

}


\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Event sequences}
