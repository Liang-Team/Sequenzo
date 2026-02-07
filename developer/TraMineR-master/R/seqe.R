## ========================================
## Check if subsequence
## ========================================
is.eseq <- function(eseq, s) {
  TraMineR.check.depr.args(alist(eseq = s))
#   return(.Call(C_istmrsequence,eseq))
    return(inherits(eseq,"eseq"))
}
is.seqelist <- function(eseq, s) {
  TraMineR.check.depr.args(alist(eseq = s))
#   return(.Call(C_istmrsequence,eseq))
  return(inherits(eseq,"seqelist"))
}
#as.seqelist<-function(eseq){
#   return(.Call(C_istmrsequence,eseq))
#    class(eseq)<-c("seqelist","list")
#    return(eseq)
#}

###Methods taken from survival package

"[.seqelist" <- function(x, i,j, drop=FALSE) {
    # If only 1 subscript is given, the result will still be a Surv object
    #  If the second is given extract the relevant columns as a matrix
  if (missing(j)) {
  	temp <- class(x)
  	type <- attr(x, "type")
  	class(x) <- NULL
  	x <- x[i, drop=FALSE]
  	class(x) <- temp
  	attr(x, "type") <- type
  	x
	} else {
  	class(x) <- NULL
  	NextMethod("[")
	}
}
Math.seqelist <- function(...){
  stop("Invalid operation on event sequences")
}
Ops.seqelist  <- function(...){
  stop("Invalid operation on event sequences")
}
Summary.seqelist<-function(...) {
  stop("Invalid operation on event sequences")
}
Math.eseq <- function(...)  {
  stop("Invalid operation on event sequences")
}
Ops.eseq  <- function(...)  {
  stop("Invalid operation on event sequences")
}
Summary.eseq<-function(...) {
  stop("Invalid operation on event sequences")
}

levels.eseq<-function(x,...){
  if(!is.eseq(x))stop("x should be a eseq object. See help on seqecreate.")
  return(.Call(C_tmrsequencegetdictionary,x))
}

levels.seqelist<-function(x,...){
  if(!is.seqelist(x))stop("x should be a seqelist. See help on seqecreate.")
  if(length(x)>0) return(.Call(C_tmrsequencegetdictionary,x[[1]]))
}
## ========================================
## Return a string representation of a sequence
## ========================================

str.seqelist<-function(object,...){
#message("Event sequence analysis module is still experimental")
  if(is.seqelist(object)){
      object<-cat(as.character(object),"\n")
  }else if (is.eseq(object)){
    object<-cat(as.character(object),"\n")
  }else{
    stop("object should be a seqelist. See help on seqecreate.")
  }
  NextMethod("str")
}
str.eseq<-function(object,...){
#  seqestr(eseq)
  if(!is.eseq(object))stop("object should be a eseq object. See help on seqecreate.")
  object <- .Call(C_tmrsequencestring, object)
  NextMethod("str")
}

as.character.eseq<-function(x, ...){
#  seqestr(eseq)
  if(!is.eseq(x))stop("x should be a eseq object. See help on seqecreate.")
  x<-.Call(C_tmrsequencestring,x)
  NextMethod("as.character")
}

as.character.seqelist<-function(x, ...){
  tmrsequencestring.internal<-function(eseq){
    if(is.eseq(eseq)){
      return(.Call(C_tmrsequencestring, eseq))
    }
    return(as.character(eseq))
  }
  if(!is.seqelist(x))  stop("x should be a seqelist object. See help on seqecreate.")
  x <- as.character(sapply(unlist(x), tmrsequencestring.internal))
  NextMethod("as.character")
}

## ========================================
## Print sequences
## ========================================

#seqeprint<-function(eseq){
#  print(seqestr(eseq))
#}
print.eseq<-function(x,quote = FALSE, ...){
  x <- as.character(x)
  print(x, quote=quote, ...)
}
print.seqelist<-function(x, quote = FALSE, ...){
  x <- as.character(x)
  print(x, quote=quote, ...)
}

## ========================================
## Plot sequences
## ========================================

plot.eseq <- function(x, type = "pc", ...) {
  if (type == "pc") {
    seqpcplot(x, ...)
  }
}

plot.seqelist <- function(x, type = "pc", ...) {
  if (type == "pc") {
    seqpcplot(x, ...)
  }
}
