seqedist <- function(seqe, idcost, vparam, interval="No", norm="YujianBo"){
	if (!is.seqelist(seqe)) {
		stop(" [!] seqe should be a seqelist. See help on seqecreate.")
	}
	if(length(idcost)!=length(levels(seqe))){
		stop(" [!] You should specify one indel cost per event (get the list of events with levels(seqe)). ")
	}
	
	interval <- pmatch(tolower(interval), c("no", "", "previous", "next"))
	if(!(is.numeric(interval) && interval %in% c(1, 3:4))){
		stop(" [!] Unknown value for interval. It should be one of: 'No', 'previous' or 'next'.")
	}
	interval <- as.integer(interval-1)
	norm <- pmatch(tolower(norm), c("none", "max", "yujianbo"))
	if(!(is.numeric(norm) && norm %in% 1:3)){
		stop(" [!] unknown normalization method ", norm, ".  It should be one of: 'none', 'max' or 'YujianBo' (default).")
	}
	norm <- as.integer(norm-1)
  TraMineR:::seqedist(seqe, idcost=as.double(idcost), vparam=as.double(vparam), norm=norm, interval=interval);
	## seqedist <- function(seqe, idcost, vparam, interval=TRUE, norm=TRUE)
}
