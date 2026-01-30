## To be used with seqprecstart and seqprecorr
states.check <- function(seqdata, state.order, state.equiv, with.missing=FALSE){


  if(length(state.order) != length(unique(state.order)))
      msg.stop("Multiple occurrences of same state in state.order: ", paste(state.order, collapse=" "))

  alphabet <- alphabet(seqdata, with.missing)
  inexistant_al <- which(is.na(match(state.order, alphabet)))
  ## Check that the listed inexistant state is not NA
  if(length(inexistant_al)>0 && !is.numeric(seqdata)) {
    if(length(inexistant_al)>1 || !is.na(state.order[inexistant_al])) {
      msg.stop("Bad state.order, states not in the alphabet: ", paste(state.order[inexistant_al], collapse=" "))
    }
  }

  if (!is.null(state.equiv)){
    if(!is.list(state.equiv)){
      msg.stop("Bad state.equiv. A list is expected!")
    }
    equiv_al <- unlist(state.equiv)
    if(length(equiv_al) != length(unique(equiv_al)))
        msg.stop("Multiple occurrence of same state in state.equiv")
    inexistant_al <- which(is.na(match(equiv_al, alphabet)))
    ## Check that the listed inexistant state is not NA
    if(length(inexistant_al)>0 && !is.numeric(seqdata)) {
      if(length(inexistant_al)>1 || !is.na(equiv_al[inexistant_al])) {
        msg.stop("Bad state.equiv, states not in the alphabet: ", paste(equiv_al[inexistant_al], collapse=" "))
      }
    }

    ### Should check that states in equiv class are contiguous in state.order
    ###  and that they contain at least one element of state.order

    ## When equiv.class contain both non-comparable and comparable states
    ## Changing status of uncomparable in the class to comparable

### ## When equiv.class include states not in the state order
### ## we add them to state.order next to the first valid element of the equiv class
###
    if (length(unique(state.order)) < length(alphabet)){
      inoncomp <- which(is.na(match(alphabet(seqdata),unique(state.order))))
      state.noncomp <- alphabet(seqdata)[inoncomp]
      ii.noncomp.equiv <- match(state.noncomp,equiv_al)
      ii.noncomp.equiv <- ii.noncomp.equiv[!is.na(ii.noncomp.equiv)]
      if(length(ii.noncomp.equiv)>0){
        state.noncomp.equiv <- equiv_al[ii.noncomp.equiv]
  ###
        ## In case a non comparable state belongs to an equiv class
        ## we add it to state.order next to the first valid element of the equiv class
        ##
        for (i in 1:length(state.noncomp.equiv)){
          for (k in 1:length(state.equiv)){
            if (state.noncomp.equiv[i] %in% state.equiv[[k]] ){
              ## insert the equiv state next to first state of the class in state.order
              ii <- match(state.equiv[[k]],state.order)
              if (!is.na(ii[1])){
                state.order.new <- c(state.order[1:ii[1]],state.noncomp.equiv[i])
                if (length(state.order)>ii[1]) {
                  state.order.new <- c(state.order.new, state.order[(ii[1]+1):length(state.order)])
                }
                state.order <- state.order.new
              } else {} ## only non comparable state in state.equiv[[k]]
            break ## we have found the corresponding state.equiv
            }
          }
        }
      }
    }
  }
###
  return(state.order)
}
