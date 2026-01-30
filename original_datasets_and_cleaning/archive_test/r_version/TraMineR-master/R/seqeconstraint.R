seqeconstraint <- function(max.gap = -1, window.size = -1, age.min = -1,
  age.max = -1, age.max.end = -1, count.method = 1, maxGap, windowSize, ageMin,
  ageMax, ageMaxEnd, countMethod) {

  TraMineR.check.depr.args(alist(max.gap = maxGap, window.size = windowSize, age.min = ageMin,
    age.max = ageMax, age.max.end = ageMaxEnd, count.method = countMethod))

	## check that all constraints are coherent
	if(age.max.end != -1 && age.max == -1){
		age.max <- age.max.end
	}
	if(max.gap!= -1 && window.size!=-1 && max.gap>window.size){
		stop(" [!] max.gap is greater than window.size")
	}
	if(age.min!= -1 && age.max!=-1 && age.min>age.max){
		stop(" [!] age.min is greater than age.max or age.max.end")
	}
        if(!count.method%in%seq(1,6,1)&
           !count.method%in%c("COBJ","CDIST_O","CWIN","CMINWIN",
                             "CDIST"))
          {
            stop(" [!] unknown count.method input")
          }
	ret <- list()
	ret$max.gap <- max.gap
	ret$window.size <- window.size
	ret$age.min <- age.min
	ret$age.max <- age.max
	ret$age.max.end <- age.max.end
        if (is.character(count.method))
          {
            ret$count.method <- switch(count.method,
                                      COBJ=1,CDIST_O=2,CWIN=3,
                                      CMINWIN=4,CDIST=5)
          } else {
            ret$count.method <- count.method
          }
	class(ret) <- "seqeconstraint"
	return(ret)
}

print.seqeconstraint<-function(x, ...){
	z<-data.frame(Constraint=names(x),Value=as.numeric(x))
	z <- z[z$"Value"!=-1, ]
        if (z[z$"Constraint"=="count.method","Value"] == 1) {
          z[z$"Constraint"=="count.method","Value"] <-
            "COBJ"
	}
	if (z[z$"Constraint"=="count.method","Value"] == 2) {
          z[z$"Constraint"=="count.method","Value"] <-
            "CDIST_0"
	}
        if (z[z$"Constraint"=="count.method","Value"] == 3) {
          z[z$"Constraint"=="count.method","Value"] <-
            "CWIN"
	}
        if (z[z$"Constraint"=="count.method","Value"] == 4) {
          z[z$"Constraint"=="count.method","Value"] <-
            "CMINWIN"
	}
        if (z[z$"Constraint"=="count.method","Value"] == 5) {
          z[z$"Constraint"=="count.method","Value"] <-
            "CDIST"
	}
	if(nrow(z) > 0) {
		print(z, row.names=FALSE,...)
	}
}
