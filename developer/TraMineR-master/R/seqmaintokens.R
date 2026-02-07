seqmaintokens <- function(seqdata, k=8L, mint=NULL, ...) {

    if (!inherits(seqdata,"stslist")){
		stop("data is NOT a state sequence object, see seqdef function to create one",
            call. = FALSE)
	}
    if (!k>=1){
        stop("k must be a strictly positive integer!",
            call. = FALSE)
    }

    meant <- seqmeant(seqdata, ...)
    if (!is.null(mint)){
        main.tokens <- which(meant >= mint)
        if (length(main.tokens) > k){
            meant.o <- order(meant[main.tokens], decreasing=TRUE)
            main.tokens <- sort(meant.o[1:k])
        }
        else if (length(main.tokens) < 1)
            message(" !!! No token occurs more than mint times on average!")
    } else {
        meant.o <- order(meant, decreasing=TRUE)
        main.tokens <- sort(meant.o[1:k])
    }
    return(main.tokens)
}
