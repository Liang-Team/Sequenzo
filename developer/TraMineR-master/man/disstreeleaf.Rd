\name{disstreeleaf}
\alias{disstreeleaf}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Terminal node membership}
\description{
  Get the terminal node membership of each case.
}
\usage{
disstreeleaf(tree, label=FALSE, collapse=", ")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tree}{A tree (\code{disstree} or \code{DissTreeNode} object).}
  \item{label}{Logical. Should leaf memberships be labelled with classification rules?}
  \item{collapse}{Character string. Separator between categories in class of categorical values.}
}
\details{
\code{disstreeleaf} returns the terminal node membership of the cases either as the leaf number of the terminal node to which the cases are assigned or, when \code{label=TRUE}, as the classification rule leading to the assigned terminal node. In the latter case, \code{collapse} is used as separator between categorical values in classes of categorical values. The default \code{collapse} is \code{", "}. It is advisable to change this default when categorical values contain commas.
}

\value{Either a vector of leaf numbers or a factor. When \code{label=FALSE} (default), vector of assigned terminal node numbers. When \code{label=TRUE}, a factor with levels labelled with classification rules.
}


\seealso{\code{\link{disstree}} and examples therein, \code{\link{disstree.get.rules}}, and \code{\link{disstree.assign}}.}
\author{Matthias Studer and Gilbert Ritschard)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
