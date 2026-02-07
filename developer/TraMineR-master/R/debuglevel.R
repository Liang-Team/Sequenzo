
debuglevel <- function(level=NULL) {
	if(is.null(level)){
		return(.Call(C_getTraMineRDebugLevel))
	}
	.Call(C_setTraMineRDebugLevel,as.integer(level))
	return(level)
	
}
