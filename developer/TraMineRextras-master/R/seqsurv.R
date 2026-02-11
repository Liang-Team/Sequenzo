# Authors: Pierre-Alexandre Fonta, Matthias Studer (2016), Gilbert Ritschard (2018)
seqsurv <- function(seqdata, groups = NULL, per.state = FALSE, state = NULL,
    with.missing = FALSE) {
  if (!inherits(seqdata, "stslist"))
    stop(strwrap(" [!] seqdata isn't a sequence object, use the seqdef function
      to create one"), call. = FALSE)

  # FIXME (Matthias) What to do with missing values?
  if (isTRUE(with.missing)) message(" [!] with.missing = TRUE isn't implemented yet")

  # FIXME (Pierre-Alexandre) Spell computations are done for each np because
  # of the curent design of seqplot. Waiting seqplot() redesign.
  ## GR Jan 29, 2018: Fixed in the seqsplot redesign of seqplot
  rownames(seqdata) <- as.character(1:nrow(seqdata))
  spell <- suppressMessages(seqformat(seqdata, from = "STS", to = "SPELL"))
  spell$id <- as.numeric(as.character(spell$id))
  spell$dur <- spell$end - spell$begin + 1
  spell$weights <- attr(seqdata, "weights")[spell$id]
  seq.length <- seqlength(seqdata)
  spell$length <- seq.length[spell$id]
  spell$status <- spell$end != spell$length
  ## GR states must be ordered according to the alphabet for color matches
  #stlev <- alphabet(seqdata)[alphabet(seqdata) %in% levels(spell$states)]
  #spell$states <- factor(spell$states, levels=stlev)
  spell$states <- factor(spell$states, levels=alphabet(seqdata))

  obs.states <- seqstatl(seqdata)
  if (is.null(state)) state <- obs.states
  else {
     state <- state[state %in% obs.states]
     if (length(state) < 1)
        stop(strwrap(" [!] state contains only unobserved states"), call. = FALSE)
  }
 ## Defining group related variables (colors, ltext, ...)
  if (is.null(groups))  ## Make a single group
    groups <- factor(rep("1", nrow(seqdata)))
  else  ## Use factor() even if group is already a factor to remove unused levels.
    groups <- factor(groups)
  spell$group <- groups[spell$id]

  levels.num <- nlevels(groups)
  # brewer.pal: minimal value for n is 3
  cpal <-
    if (levels.num == 1) {
      ##message(" [!] only one group, no automatic color palette assignation")
      ##NULL
      brewer.pal(3, "Dark2")[3]
    } else if (levels.num == 2) {
      brewer.pal(3, "Dark2")[-3]
    } else if (levels.num < 9) {
      brewer.pal(levels.num, "Dark2")
    } else {
      message(" [!] too many groups (> 8), no automatic color palette assignation")
      NULL
    }
  ltext <- levels(groups)

##################

#   if (isTRUE(per.state)) {
#     #if (length(state) > 1) state <- state[1]
#     if (any(spell$states %in% state)){
#       res <- survfit(Surv(spell$dur, spell$status) ~ 1, weights = spell$weights,
#         subset = spell$states %in% state)
#       ltext <- levels(factor(spell$status))
#     }
#     else res <- 0
#   }
#   else if (is.null(state)){
#       res <- survfit(Surv(spell$dur, spell$status) ~ spell$states,
#         weights = spell$weights)
#     }
#     else if (any(spell$states %in% state)){ ## want only curves of a subset of states
#       res <- survfit(Surv(spell$dur, spell$status) ~ spell$states,
#         subset = spell$states %in% state, weights = spell$weights)
#     } else { ## no valid cases
#       res <- 0
#     }
#     cpal <- cpal(seqdata)
#
# #  else {  ## !is.null(groups)

####

    if (isTRUE(per.state)) {
      if (any(spell$states %in% state)){
        res <- survfit(Surv(spell$dur, spell$status) ~ spell$group,
          weights = spell$weights, subset = spell$states %in% state)
      }
      else res <- 0  ## No valid result

    } else if (levels.num == 1){ ## per.state = FALSE and single group
      if (any(spell$states %in% state)){
        res <- survfit(Surv(spell$dur, spell$status) ~ spell$states,
        weights = spell$weights, subset = spell$states %in% state)
        ##ltext <- levels(factor(spell$status))
        ltext <- attr(seqdata, "labels")[alphabet(seqdata) %in% state]
        cpal <- attr(seqdata, "cpal")[alphabet(seqdata) %in% state]
      }
      else res <- 0

      # group != NULL && per.state == FALSE <=> groups == NULL && per.state == FALSE
      # with the current design of seqplot()
      # FIXME (Pierre-Alexandre) Waiting seqplot() redesign.

    } else { ## survfit not applicable for per.state = FALSE and more than 1 group
      stop(strwrap(" [!] With per.state = FALSE, only a single group is supported. Consider using seqsplot() instead."), call. = FALSE)
    }

  if (exists("ltext")) attr(res, "ltext") <- ltext
  attr(res, "cpal") <- cpal
  attr(res, "xtstep") <- attr(seqdata, "xtstep")
  attr(res, "tick.last") <- attr(seqdata, "tick.last")
  class(res) <- c("stslist.surv", class(res))
  return(res)
}
