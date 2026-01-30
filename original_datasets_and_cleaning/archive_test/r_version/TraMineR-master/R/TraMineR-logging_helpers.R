# Logging helpers for error, warning and information situations.
#
# Simplify and harmonize logging and the flow of control in TraMineR 2.
#
# @param msg Message to print before stopping or continuing function execution.
#
# @return None
#
# @author Pierre-Alexandre Fonta (2017)


#### Generic ####

# stop
msg.stop <- function(msg, ...) {
  stop(paste(msg, ...), call. = F)
}

# warn
f.warn <- function(f, msg, ...) {
  message(paste(" [!!]", f(msg, ...)))
}
msg.warn <- function(msg, ...) {
  f.warn(paste, msg, ...)
}
msg.warn0 <- function(msg, ...) {
  f.warn(paste0, msg, ...)
}

# info
f.info <- function(f, msg, ...) {
  message(paste(" [>]", f(msg, ...)))
}
msg <- function(msg, ...) {
  f.info(paste, msg, ...)
}
msg0 <- function(msg, ...) {
  f.info(paste0, msg, ...)
}


#### Arguments ####

# utils
aprint <- function(arg) {
  return(paste0("'", arg, "'"))
}

# not yet implemented
msg.stop.impl <- function(arg, method, values = NULL, when = NULL) {
  a <- aprint(arg)
  m <- paste0("'method = \"", method, "\"'")
  if (is.null(values)) {
    if (is.null(when))
      msg.stop(a, "isn't implemented yet for", m)
    else
      msg.stop(a, "isn't implemented yet for", m, "when", when)
  } else {
    v <- paste(values, collapse = ", ")
    msg.stop(a, "isn't implemented yet for values", v, "for", m)
  }
}

# missing
msg.stop.miss <- function(arg) {
  msg.stop(aprint(arg), "must be specified")
}

# not in
msg.stop.in <- function(arg, values) {
  msg.stop(aprint(arg), "must be one of:", paste(values, collapse = ", "))
}

# unknown
msg.stop.na <- function(arg) {
  msg.stop(aprint(arg), "invalid type or unknown value")
}

# internal error
msg.stop.ie <- function(info, ...) {
  msg.stop("internal error, contact the package maintainer:", info, ...)
}

# empty sequence
msg.stop.sempty <- function(method, values) {
  msg.stop(method," does not support empty sequences ", paste(values, collapse = ", "))
}

msg.warn.sempty <- function(values) {
  msg.warn("Empty sequence(s) ", paste(values, collapse = ", "), " may cause inconsistent results!")
}


# ignored
msg.warn.ign1 <- function(arg) {
  msg.warn(aprint(arg), "has been ignored")
}
msg.warn.ign2 <- function(iarg, sarg) {
  msg.warn(aprint(iarg), "has been ignored because", aprint(sarg), "is specified")
}
