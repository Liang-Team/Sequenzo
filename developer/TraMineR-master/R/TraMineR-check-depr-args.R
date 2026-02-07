# Function to assure a smooth transition for deprecated arguments.
#
# For each specified pair of new and old argument names, check if the old
# argument name is specified. If so but the new one not, show a warning message
# and use the value. If one of the name declared in TraMineR.check.depr.args() arguments
# doesn't exist or if the new and old argument names are specified together,
# show an error message and exit.
#
# It doesn't detect when { the new and the old argument names are specified
# together and the new argument value is its default value }. In this case,
# the value associated with the old argument name is used and the warning
# message is shown.
#
# It works with or without using argument names in the checked function.
#
# Add in the checked function parameters the old argument name WITHOUT default value.
#
# @param arg.pairs List of pairs of new and old names for each renamed argument.
#  Format: alist(new = old, ...).
#
# @return None
#
# @author Pierre-Alexandre Fonta (2016-2017)

TraMineR.check.depr.args <- function(arg.pairs) {
  new.names <- names(arg.pairs)
  old.names <- as.character(arg.pairs)
  pf <- parent.frame()
  fun.args <- ls(pf)
  for (i in 1:length(arg.pairs)) {
    new.name <- new.names[i]
    old.name <- old.names[i]
    if (new.name %in% fun.args && old.name %in% fun.args) {
      calling.fun <- paste0(as.character(sys.calls()[[1]])[1], "()")
      new.val <- eval(substitute(pf$nn, list(nn = new.name)))
      old.val <- eval(substitute(pf$on, list(on = old.name)))
      has.new <- !missing(new.val) # TRUE if new has a default value!
      has.old <- !missing(old.val)
      fun.args.default <- formals(sys.function(-1))
      new.default <- eval(substitute(fun.args.default$nn, list(nn = new.name)))
      has.new.default <- !missing(new.default)
      if (has.new.default) {
        new.default <- tryCatch(
          # For '1:10' like expressions
          eval(new.default),
          # For argument which default value is an other argument
          error = function(e) eval(substitute(pf$nd, list(nd = new.default))))
      }
      if (has.old) {
        if (has.new && (!has.new.default || (has.new.default && !identical(new.val, new.default)))) {
          msg.stop("In", calling.fun, ":", new.name, "and", old.name,
            "cannot be specified together.", old.name, "is deprecated, use",
            new.name, "instead.")
        } else {
          msg.warn("In", calling.fun, ":", old.name, "is deprecated, use", new.name, "instead.")
          assign(new.name, old.val, envir = pf)
        }
      }
    } else {
      # Don't use msg.stop() because the calling function must be displayed
      stop(new.name, " = ", old.name, " is incorrect: at least one argument doesn't exist")
    }
  }
}

######### For backward compatibility: used by TraMineRextras v 4.1

checkargs <- TraMineR.check.depr.args
