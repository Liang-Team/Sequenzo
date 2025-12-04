# Check if an object is a valid data frame or matrix index.
#
# Otherwise, an error is raised and an information message is displayed.
#
# @param x     Object to test.
#        name  Name of the object to test.
#
# @return Nothing.
#
# @author Pierre-Alexandre Fonta (2016-2017)

checkindex <- function(x, name) {
  if (!is.index(x))
    msg.stop(aprint(name), "must be a positive integer or a string")
}

# Check if an object is a vector of valid data frame or matrix indexes.
#
# Otherwise, an error is raised and an information message is displayed.
#
# @param x     Object to test.
#        name  Name of the object to test.
#
# @return Nothing.
#
# @author Pierre-Alexandre Fonta (2016-2017)

checkindexes <- function(x, name) {
  if (!is.vector(x) || !xor(is.positive.integers(x), is.strings(x)))
    msg.stop(aprint(name), "must be a vector of positive integers or strings")
}
