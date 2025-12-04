\name{TraMineR.check.depr.args}
\alias{TraMineR.check.depr.args}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Checking and managing deprecated arguments
}
\description{
Checks the presence of deprecated arguments, assigns value of a deprecated argument to the corresponding new argument name, and issues warning messages.
}
\usage{
TraMineR.check.depr.args(arg.pairs)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{arg.pairs}{List of pairs of old and new argument names
  \cr(e.g. \code{alist(newname1 = oldname1, newname2 = oldname2)})}
}

\details{To be used inside functions. For developers only.

For each specified pair of new and old argument names, the function checks if the old
argument name is specified. If so and the new one is not, a warning message is raised
and the argument value is assigned to the new argument name. If one of the names declared in check.depr.args() arguments si not an argument of the parent function
or if both the new and old argument names are specified an error is raised.

The function does not detect when the new and the old argument names are specified
together and the new argument value is its default value. In this case,
the value associated with the old argument name is assigned to the new name and a warning
message is raised.

The function works whether the argument names are explicitly declared or not in the call to the checked function.

The only requirement for the function to work is that the deprecated arguments should be listed WITHOUT default values in the definition of the checked function.
}

\author{Pierre-Alexandre Fonta, Gilbert Ritschard}

\value{None.} 