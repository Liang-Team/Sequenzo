wcPropertyClustering <- function(diss, properties, maxcluster=NULL, ...){
	form <- as.formula(paste("diss~`", paste(names(properties), collapse="`+`"), "`", sep=""))
	st <- disstree(form, data=properties, ...)
	class(st) <- c("dtclust", class(st))

	st <- clusterSplitSchedule(st)
	if(!is.null(maxcluster)){
		return(dtprune(st, maxcluster, diss))
	}
	return(st)
}

seqpropclust <- function(seqdata, diss, properties=c("state", "duration", "spell.age", 
						"spell.dur", "transition", "pattern", "AFtransition", "AFpattern", "Complexity"), 
						other.prop=NULL, prop.only=FALSE, pmin.support=0.05,  max.k = -1, 
						with.missing=TRUE, R=1, weight.permutation="diss", min.size=0.01, 
						max.depth = 5, maxcluster=NULL, ...){
	all.props <- list()
	nbseq <- nrow(seqdata)
	if(!is.null(other.prop)){
		mat <- as.data.frame(other.prop)
		message(" [>] Adding ", ncol(mat), " user defined properties.", appendLF=FALSE)
		all.props[["other.prop"]] <- mat
		if(nrow(mat)!=nbseq){
			stop(" [!] The number of row of the 'other.prop' argument should equal the number of sequences.")
		}
	}
	if("state" %in% properties){
		message(" [>] Extracting 'state' properties...", appendLF=FALSE)
		all.props[["state"]] <- as.data.frame(seqdata)
		message("OK (", ncol(all.props[["state"]]), " properties extracted)")
	}
	if("spell.dur" %in% properties ||"spell.age" %in% properties){
		spelldur <- "spell.dur" %in% properties
		spellage <- "spell.age" %in% properties
		message(" [>] Extracting 'spell' properties...", appendLF=FALSE)
		dss <- as.matrix(seqdss(seqdata, with.missing=with.missing))
		statl <- attr(seqdata, "alphabet")
		nr <- attr(seqdata, "nr")
		void <- attr(seqdata, "void")
		if (with.missing) {
			statl <- c(statl, nr)
		}
		numspell <- list()
		
		blank <- rep(0, nbseq)
		nablank <- rep(NA, nbseq)
		for(st in statl){
			num <- max(apply(dss, 1, function(x) sum(x==st)))
			if(num>0){
				for(i in 1:num){
					if(spelldur){
						numspell[[paste(st, i, sep="_dur_")]] <- blank
					}
					if(spellage){
						numspell[[paste(st, i, sep="_age_")]] <- nablank
					}
				}
			}
		}
		spellprop <- do.call(data.frame, numspell)
		seqduree <- seqdur(seqdata, with.missing=with.missing)
		maxsp <- ncol(dss)
		# print(head(dss))
		# print(head(seqduree))
		# print(summary(spellprop))
		# print(nrow(spellprop))
		# print(maxsp)
		for(i in 1:nbseq){
			j <- 1
			## cat("i=", i, "\n")
			age <- 0
			while(j<=maxsp && dss[i, j]!=void){
				st <- dss[i, j]
				stnum <- sum(dss[i, 1:j]==st)
				## cat("    j=", j, st, sp, "\n")
				if(spelldur) spellprop[i, paste(st, stnum, sep="_dur_")] <- seqduree[i, j]
				if(spellage) spellprop[i, paste(st, stnum, sep="_age_")] <- age
				age <- age +  seqduree[i, j]
				## cat("    j=", j, st, sp, seqduree[i, j],"\n")
				j <- j + 1
			}
		}
		all.props[["spell"]] <- as.data.frame(spellprop)
		message("OK (", ncol(all.props[["spell"]]), " properties extracted)")
	}
	if("duration" %in% properties){
		message(" [>] Extracting 'duration' properties...", appendLF=FALSE)
		suppressMessages(all.props[["duration"]] <- as.data.frame(seqistatd(seqdata, with.missing=with.missing)))
		message("OK (", ncol(all.props[["duration"]]), " properties extracted)")
	}
	if("transition" %in% properties){
		message(" [>] Extracting 'transition' properties...", appendLF=FALSE)
		seqe <- seqecreate(seqdata, tevent="transition")
		fsub <- seqefsub(seqe, pmin.support=pmin.support, max.k=max.k)
		mat <- as.data.frame(seqeapplysub(fsub, method="count"))
		row.names(mat) <- NULL
		all.props[["transition"]] <- mat
		message("OK (", ncol(all.props[["transition"]]), " properties extracted)")
	}
	if("pattern" %in% properties){
		message(" [>] Extracting 'pattern' properties...", appendLF=FALSE)
		seqe <- seqecreate(seqdata, tevent="state")
		fsub <- seqefsub(seqe, pmin.support=pmin.support, max.k=max.k)
		mat <- as.data.frame(seqeapplysub(fsub, method="count"))
		row.names(mat) <- NULL
		all.props[["pattern"]] <- mat
		message("OK (", ncol(all.props[["pattern"]]), " properties extracted)")
	}
	if("AFtransition" %in% properties){
		message(" [>] Extracting 'AFtransition' properties...", appendLF=FALSE)
		seqe <- seqecreate(seqdata, tevent="transition")
		fsub <- seqefsub(seqe, pmin.support=pmin.support, max.k=max.k)
		mat <- seqeapplysub(fsub, method="age")
		mat[mat == -1] <- NA
		mat <- as.data.frame(mat)
		row.names(mat) <- NULL
		all.props[["AFtransition"]] <- mat
		message("OK (", ncol(all.props[["AFtransition"]]), " properties extracted)")
	}
	if("AFpattern" %in% properties){
		message(" [>] Extracting 'AFpattern' properties...", appendLF=FALSE)
		seqe <- seqecreate(seqdata, tevent="state")
		fsub <- seqefsub(seqe, pmin.support=pmin.support, max.k=max.k)
		mat <- seqeapplysub(fsub, method="age")
		mat[mat == -1] <- NA
		mat <- as.data.frame(mat)
		row.names(mat) <- NULL
		all.props[["AFpattern"]] <- mat
		message("OK (", ncol(all.props[["AFpattern"]]), " properties extracted)")
	}
	if("Complexity" %in% properties){
		message(" [>] Extracting 'Complexity' properties...", appendLF=FALSE)
		suppressMessages(mat <- data.frame(ici=seqici(seqdata), ient=seqient(seqdata), turb=seqST(seqdata), trans=seqtransn(seqdata)))
		all.props[["Complexity"]] <- mat
		message("OK (", ncol(all.props[["Complexity"]]), " properties extracted)")
	}
	if(any(sapply(all.props, nrow)!=nbseq)){
		stop(" [!] Feature extraction failed.")
	}
	if(length(all.props)>1){
		properties <- do.call(cbind, all.props)
	}else{
		properties <- all.props[[1]]
	}
	message(" [>] ", ncol(properties), " properties extracted.")
	if(prop.only){
		return(properties)
	}
	form <- as.formula(paste("seqdata~`", paste(names(properties), collapse="`+`"), "`", sep=""))
	st <- seqtree(form, data=properties, diss=diss, R=R, 
				weight.permutation=weight.permutation, min.size=min.size, max.depth = max.depth, ...)
	class(st) <- c("seqtreeclust", "dtclust", class(st))
	st <- clusterSplitSchedule(st)
	if(!is.null(maxcluster)){
		return(dtprune(st, maxcluster, diss))
	}
	return(st)
}

clusterSplitSchedule <- function(tree) {
	treeEnv <- environment()
	treeEnv$trsize <- 0
	treeEnv$numsplit <- 0
	treeSize <- function(node){
		treeEnv$trsize <- treeEnv$trsize+1
		if (!is.null(node$kids)) {
			treeEnv$numsplit <- treeEnv$numsplit+1
			treeSize(node$kids[[1]])
			treeSize(node$kids[[2]])
		}
	}
	treeSize(tree$root)
	resetSplitSchedule <- function(node){
		node$info$splitschedule <- 0
		if (!is.null(node$kids)) {
			node$kids[[1]] <- resetSplitSchedule(node$kids[[1]])
			node$kids[[2]] <- resetSplitSchedule(node$kids[[2]])
		}
		return(node)
	}
	setSplitSchedule <- function(node, sp, ids){
		if(node$id %in% ids){
			node$info$splitschedule <- sp
		}
		else if (!is.null(node$kids)) {
			node$kids[[1]] <- setSplitSchedule(node$kids[[1]], sp, ids)
			node$kids[[2]] <- setSplitSchedule(node$kids[[2]], sp, ids)
		}
		return(node)
	}
	findSplitSchedule <- function(node){
		if (!is.null(node$kids)){
			if(node$kids[[1]]$info$splitschedule==0){
				SCtot <- node$info$vardis*node$info$n
				SCexpl <- node$split$info$R2*SCtot
				treeEnv$SCexpl[as.character(node$id)] <- SCexpl
				treeEnv$kids[[as.character(node$id)]] <- c(node$kids[[1]]$id, node$kids[[2]]$id)
			} else{
				findSplitSchedule(node$kids[[1]])
				findSplitSchedule(node$kids[[2]])
			}
		}
	}
	tree$root <- resetSplitSchedule(tree$root)
	tree$root$info$splitschedule <- 1
	
	for(sp in 2:(treeEnv$numsplit+1)){
		treeEnv$SCexpl <- numeric()
		treeEnv$kids <- list()
		findSplitSchedule(tree$root)
		oo <- order(treeEnv$SCexpl, decreasing=TRUE)
		# print(names(treeEnv$SCexpl)[oo])
		# print(treeEnv$SCexpl[oo])
		# print(treeEnv$kids)
		id <- names(treeEnv$SCexpl)[oo[1]]
		tree$root <- setSplitSchedule(tree$root, sp, treeEnv$kids[[id]])
	}
	return(tree)
}
dtcut <- function(st, k, labels=TRUE){
	max.k <- length(unique(st$fitted[, 1]))
	if(k>max.k){
		stop(" [!] The maximum number of groups is ", max.k)
	}
	treeEnv <- environment()
	treeEnv$base <- rep(st$root$id, length(st$fitted[, 1]))
	if(k==1){
		return(treeEnv$base)
	}
	getleaf <- function(node) {
		if (node$info$splitschedule <= k) {
			treeEnv$base[node$info$ind] <- node$id
		} 
		if (!is.null(node$kids)) {
			getleaf(node$kids[[1]])
			getleaf(node$kids[[2]])
		}
		
	}
	getleaf(st$root)
	if(labels){
		lab <- dtlabels(st)
		return(factor(factor(treeEnv$base, levels=as.numeric(names(lab)), labels=lab)))
	}
	return(treeEnv$base)
}
dtlabels <- function(tree){
	if (!inherits(tree, "disstree")) {
		stop("tree should be a disstree object")
	}

	split_s <- function(sp){
		formd <- function (x){
			return(format(x, digits =getOption("digits")-2))
		}
		str_split <- character(2)
		vname <- colnames(tree$data)[sp$varindex]
		if (!is.null(sp$breaks)) {
			str_split[1] <- paste("<=", formd(sp$breaks))
			str_split[2] <- paste(">", formd(sp$breaks))
		}
		else {
			str_split[1] <- paste0("[", paste(sp$labels[sp$index==1], collapse=", "),"]")
			str_split[2] <- paste0("[", paste(sp$labels[sp$index==2], collapse=", "),"]")
		}
		if(!is.null(sp$naGroup)){
			str_split[sp$naGroup] <- paste(str_split[sp$naGroup], "with NA")
		}
		return(paste(vname, str_split))
	}
	labelEnv <- new.env()
	labelEnv$label <- list()
	addLabel <- function(n1, n2, val){
		id1 <- as.character(n1$id)
		id2 <- as.character(n2$id)
		labelEnv$label[[id2]] <- c(labelEnv$label[[id1]], val)
	}
	nodeRec <- function(node){
		if(!is.null(node$split)){
			spl <- split_s(node$split)
			addLabel(node, node$kids[[1]], spl[1])
			addLabel(node, node$kids[[2]], spl[2])
			nodeRec(node$kids[[1]])
			nodeRec(node$kids[[2]])
		}
	}
	nodeRec(tree$root)
	l2 <- list()
	for(nn in names(labelEnv$label)){
		l2[[nn]] <- paste0(labelEnv$label[[nn]], collapse=" & ")
	}
	l3 <- as.character(l2)
	names(l3) <- names(l2)
	return(l3)
}
dtprune <- function(st, k, diss){
	prune <- function(node) {
		if (!is.null(node$kids)){
			if(node$kids[[1]]$info$splitschedule>k) {
				node$split <- NULL
				node$kids <- NULL
			}else{
				node$kids[[1]] <- prune(node$kids[[1]])
				node$kids[[2]] <- prune(node$kids[[2]])
			}
		}
		return(node)
	}
	st$root <- prune(st$root)
    st$fitted[, 1] <- disstreeleaf(st$root)

    st$info$prune <- k
	if(is.null(diss)){
		st$info$adjustment <- NULL
	}else{
		if (st$info$weight.permutation == "none") {
			st$info$adjustment <- dissassoc(diss, st$fitted[, 1], R = st$info$R, weights = NULL)
		}
		else {
			st$info$adjustment <- dissassoc(diss, st$fitted[, 1], R = st$info$R, 
			weights = st$info$weights, weight.permutation = st$info$weight.permutation)
		}
	}
    return(st)
}

as.clustrange.dtclust <- function(object, diss, weights=NULL, R=1,  samplesize=NULL, ncluster=20, labels=TRUE, ...){
	if(ncluster<3){
		stop(" [!] ncluster should be greater than 2.")
	}
	
    max.k <- length(unique(object$fitted[, 1]))
	if(ncluster > max.k){
		stop(" [!] ncluster should be less than ", max.k+1)
	}
	lab <- dtlabels(object)
	pred <- data.frame(Split2=dtcut(object, 2, labels))
	for(p in 3:ncluster){
		pred[, paste("Split", p, sep="")] <- dtcut(object, p, labels)
	}
	object <- pred
	as.clustrange(object, diss=diss, weights=weights, R=R, samplesize=samplesize, ...)
}


dfdisplay <- function(tree, data=NULL, ...){
	data.props <- function(df){
		ret <- list()
		allcolumns <- 1:ncol(df)
		ret$ylims <- lapply(allcolumns, function(i){ if(!is.factor(df[, i])) range(df[, i],  na.rm = TRUE, finite=TRUE) else 0})
		ret$col.levels <- lapply(allcolumns, function(i){ if(is.factor(df[, i])) brewer.pal(nlevels(df[, i]), "Set3") else ""})
		numeric.cols <- character(ncol(df))
		nbnum <- sapply(allcolumns, function(i) is.numeric(df[, i]))
		ret$numeric.cols[nbnum] <- brewer.pal(sum(nbnum), "Dark2")
		return(ret)
	}
	if(is.null(data)){
		if(is.null(tree$object)){
			stop(" [!] You should specify either the data argument or the object argument in disstree.")
		}
		data <- as.data.frame(tree$object)
	}else{
		data <- as.data.frame(data)
	}
	myplot <- function(ind, vname=FALSE, ...){
		df <- data[ind, ]
		nc <- ncol(df)
		
		par(mfrow=c(1, nc), mar=c(0.5,0.5,ifelse(vname, 2, 0.5),0.5))
		
		for(i in 1:nc){
			main <- NULL
			if(vname) main <- names(df)[i]
			if(is.numeric(df[, i])){
				boxplot(df[, i], ylim=dp$ylims[[i]], axes=FALSE, col=dp$numeric.cols[i], main=main)
			}else{
				tt <- prop.table(table(df[, i]))
				barplot(as.matrix(tt, ncol=1), beside=FALSE, col=dp$col.levels[[i]], axes=FALSE, main=main)
			}
		}
	}
	dp <- data.props(data)
	disstreedisplay(tree, imagedata=NULL, imagefunc=myplot, title.outer=TRUE,  ...)
}


plot.seqtree <- function(x, ...){
	seqtreedisplay(tree=x, ...)
}

plot.disstree <- function(x, type="df", ...){
	if(type=="df"){
		dfdisplay(tree=x, ...)
	}else{
		disstreedisplay(tree=x, ...)
	}
}

plot.dtclust <- function(x, data=NULL, ncluster=NULL, diss=NULL, withquality=TRUE, ...){
	if(!is.null(ncluster)){
		x <- dtprune(x, ncluster, diss)
	}
	if(is.null(diss)){
		withquality <- FALSE
	}
	dfdisplay(tree=x, data=data, withquality=withquality, ...)
}

plot.seqtreeclust <- function(x, ncluster=NULL, diss=NULL, withquality=TRUE, ...){
	if(!is.null(ncluster)){
		x <- dtprune(x, ncluster, diss)
	}
	if(is.null(diss)){
		withquality <- FALSE
	}
	seqtreedisplay(tree=x, withquality=withquality, ...)
}
