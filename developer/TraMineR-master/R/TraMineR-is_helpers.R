# Check if an object is a number.
#
# @param x Object to test.
#
# @return TRUE if the object is a number, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017), Gilbert Ritschard (2020)

## GR: Check if an object is an stslist object

is.stslist <- function(x) {
  w <- attr(x,"weights")
  ret <- inherits(x,"stslist") && (is.null(w) | length(w) == nrow(x)) &&
            all(sapply(as.list(x)[1:ncol(x)],is.factor))
  if (ret && length(w) == nrow(x)) {
    ret <- length(names(w))==length(w) && all(names(w) == rownames(x))
  }
  return(ret)
}

# Check if number

is.a.number <- function(x) {
  return(is.numeric(x) && length(x) == 1)
}

# Check if an object is a positive integer.
#
# @param x Object to test.
#
# @return TRUE if the object is a positive integer, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.a.positive.integer <- function(x) {
  return(is.a.number(x) && x >= 0 && x %% 1 == 0)
}

# Check if an object is a vector of positive integers.
#
# @param x Object to test.
#
# @return TRUE if the object is a vector of positive integers, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.positive.integers <- function(x) {
  return(all(sapply(x, is.a.positive.integer, USE.NAMES = FALSE)))
}

# Check if an object is a (unique) character.
#
# @param x Object to test.
#
# @return TRUE if the object is a (unique) character, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.a.character <- function(x) {
  return(is.a.string(x) && nchar(x) == 1)
}

# Check if an object is a string.
#
# @param x Object to test.
#
# @return TRUE if the object is a string, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.a.string <- function(x) {
  return(is.character(x) && length(x) == 1)
}

# Check if an object is a vector of strings.
#
# @param x Object to test.
#
# @return TRUE if the object is a vector of strings, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.strings <- function(x) {
  return(all(sapply(x, is.a.string, USE.NAMES = FALSE)))
}

# Check if an object is a positive integer or a string.
#
# Used to verify if an object is a valid data frame or matrix index.
#
# @param x Object to test.
#
# @return TRUE if the object is a positive integer or a string, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.index <- function(x) {
  return(is.a.positive.integer(x) || is.a.string(x))
}

# Check if an object is a vector of positive integers or strings.
#
# Used to verify if a vector contains valid data frame or matrix indexes.
#
# @param x Object to test.
#
# @return TRUE if the object is a vector of positive integers or strings, FALSE otherwise.
#
# @author Pierre-Alexandre Fonta (2016-2017)

is.indexes <- function(x) {
  return(xor(is.positive.integers(x), is.strings(x)))
}
