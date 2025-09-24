
membershipnames <- function(ff, seqdata){
	clustering <- ff$membership
	colnames(clustering) <- sapply(1:ncol(clustering), function(i) suppressMessages(seqformat(seqdata[which.max(clustering[, i]), ], from = "STS", to = "SPS", compressed = TRUE))[1,])
	return(clustering)
}


crispness <- function(ff, norm=TRUE){
	if(inherits(ff, "fanny")){
		ff <- ff$membership
	}
	uu <- rowSums(ff^2)
	if(norm){
		uu <- (uu-1/ncol(ff))/(1-1/ncol(ff))
	}
	return(uu)
}

fuzzyseqplot <- function(seqdata, group=NULL, membership.threashold=0, type="i", members.weighted=TRUE, memb.exp=1, ...){
	
	if(is.null(group)||is.factor(group)){
		seqplot(seqdata, group=group, ...)
	}
	if(inherits(group, "fanny")){
		group <- group$membership
	}
	group <- group^memb.exp
	if(!is.matrix(group)||nrow(group)!=nrow(seqdata)){
		stop(" [!] The group argument should be a membership matrix with one row per sequence.")
	}
	## Expand seqdata to have one observation per individual-group
	seqdata <- seqdata[rep(1:nrow(seqdata), ncol(group)), ]
	ww <- attr(seqdata, "weights")
	if(!is.null(ww)&& members.weighted){
		attr(seqdata, "weights") <- ww*as.vector(group)
	}else{
		attr(seqdata, "weights") <- as.vector(group)
	}
	groupnames <- 1:ncol(group)
	if(!is.null(colnames(group))){
		groupnames <- colnames(group)
	}
	clustering <- rep(groupnames, each=nrow(group))
	cond <- as.vector(group) >= membership.threashold
	seqdata <- seqdata[cond, ]
	clustering <- clustering[cond]
	args <- list(seqdata=seqdata, group=clustering, type=type, ...)
	sortv <- args[["sortv"]]
	if(!is.null(sortv) && length(sortv)==1 && sortv=="membership"){
		sortv <- as.vector(group)[cond]
		args$sortv <- sortv
	}
	do.call(seqplot, args)
}



fuzzyseqplotsingle <- function(seqdata, group=NULL, level=NULL, membership.threashold=0, type="i", members.weighted=TRUE, memb.exp=1, ...){
	if(is.null(level)){
		stop(" [!] A level should be set.")
	}
	if(is.null(group)||is.factor(group)){
		seqplot(seqdata, group=group, ...)
	}
	if(inherits(group, "fanny")){
		group <- group$membership
	}
	group <- group^memb.exp
	if(!is.matrix(group)||nrow(group)!=nrow(seqdata)){
		stop(" [!] The group argument should be a membership matrix with one row per sequence.")
	}
	## Expand seqdata to have one observation per individual-group
	seqdata <- seqdata[rep(1:nrow(seqdata), ncol(group)), ]
	ww <- attr(seqdata, "weights")
	if(!is.null(ww)&& members.weighted){
		attr(seqdata, "weights") <- ww*as.vector(group)
	}else{
		attr(seqdata, "weights") <- as.vector(group)
	}
	groupnames <- 1:ncol(group)
	if(!is.null(colnames(group))){
		groupnames <- colnames(group)
	}
	clustering <- rep(groupnames, each=nrow(group))
	cond <- as.vector(group) >= membership.threashold & clustering==level
	if(!any(cond)){
		stop(" [!] no sequence to plot.")
	}
	seqdata <- seqdata[cond, ]
	args <- list(seqdata=seqdata, type=type, with.legend=FALSE, use.layout=FALSE, ...)
	sortv <- args[["sortv"]]
	if(!is.null(sortv) && length(sortv)==1 && sortv=="membership"){
		sortv <- as.vector(group)[cond]
		args$sortv <- sortv
	}
	do.call(seqplot, args)
}