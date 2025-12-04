extract.ver <- function(x) {

	ver.list <- strsplit(x, "\\.")
	ver.unit <- ver.list[[1]][1]
	if(length(grep("-", ver.list[[1]][2]))>0) {
		ver.dec.list <- strsplit(ver.list[[1]][2], "-")
		ver.dec <- ver.dec.list[[1]][1]
		ver.bug <- ver.dec.list[[1]][2]
		return(c(ver.unit, ver.dec, ver.bug))
	}

	ver.dec <- ver.list[[1]][2]
	return(c(ver.unit, ver.dec))
}
