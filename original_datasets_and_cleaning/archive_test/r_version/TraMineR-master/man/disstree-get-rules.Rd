\name{disstree.get.rules}
\alias{disstree.get.rules}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Tree classification rules}
\description{
  Tree classification rules.
}
\usage{
disstree.get.rules(tree, collapse="; ")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tree}{A tree (\code{disstree} or \code{DissTreeNode} object).}
  \item{collapse}{Character string. Separator between categories in class of categorical values.}
}
\details{
\code{disstree.get.rules} extracts the classification rules defined by a tree grown from a dissimilarity matrix and returns them as a vector of character strings. The rules are expressed as R commands and the i-th rule, for example, can be applied using \code{\link{eval}(parse(text=rule[i]))}. Rules are built through a call to \code{\link{disstreeleaf}}.
}

\value{Character vector with the rules as R commands and an attribute \code{covariates} providing the names of the variables involved in the rules.
}


\seealso{\code{\link{disstree}}, \code{\link{disstreeleaf}}, \code{\link{disstree.assign}}}
\author{Gilbert Ritschard)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
