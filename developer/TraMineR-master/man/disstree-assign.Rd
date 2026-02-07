\name{disstree.assign}
\alias{disstree.assign}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Assign rules to profiles provided}
\description{
  Find the tree classification rule that applies to provided cases.
}
\usage{
disstree.assign(rules, profile, covar=attr(rules,"covariates"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rules}{Character vector. List of classification rules such as those returned by \code{\link{disstree.get.rules}}.}
  \item{profile}{Data frame. Profiles of cases to be classified with the rules.}
  \item{covar}{Character vector. List of names of covariates used by the rules.}
}
\details{
\code{rules} must be given as strings of R commands. Use \code{\link{disstree.get.rules}} to get the classification rules of a tree of class \code{disstree}.

Rules are expected to create a full partition of the space of possible values of the covariates, i.e., any profile must satisfy one and only one of the rules.

}

\value{A vector of length equal to the number of rows of \code{profile} with for each case the index of the rule that applies.
}


\seealso{\code{\link{disstree}} and examples therein, \code{\link{disstreeleaf}}, \code{\link{disstree.get.rules}}}
\author{Gilbert Ritschard)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
