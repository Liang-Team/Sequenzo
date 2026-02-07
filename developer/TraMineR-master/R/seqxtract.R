## =================================
## EXTRACT SEQUENCES FROM A DATA SET
## =================================

seqxtract <- function(data, var, data.frame=FALSE) {

    ## tibble transformed into data frame
    if (inherits(data, "tbl_df")) data <- as.data.frame(data) 	
	## Extracting the sequences from the data set
	if (missing(var) || is.null(var) || is.na(var[1]))
		seqdata <- data
	else
		seqdata <- subset(data,,var)

	if (data.frame==FALSE)
		seqdata <- as.matrix(seqdata)

	return(seqdata)

	}
