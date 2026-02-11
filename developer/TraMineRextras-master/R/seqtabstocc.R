# Frequencies of state co-occurrence patterns

seqtabstocc <- function(seqdata, with.missing=FALSE, ...){
    m <- as.matrix(seqdata)
    alphabet <- alphabet(seqdata)
    misschar <- attr(seqdata, 'nr')
    if (with.missing){
      misschar <- attr(seqdata, 'void')
      alphabet <- c(alphabet,attr(seqdata,'nr'))
    }
    else{
      m[m==attr(seqdata,'void')] <- misschar
    }
    ssort <- t(apply(m, MARGIN=1, sort))
    ssort.seq <- suppressMessages(seqdef(ssort, missing=misschar, nr=paste0(misschar,attr(seqdata,'nr')))) 
    ##matchlab <- match(alphabet(ssort.seq),alphabet)
    ##ssort.seq <- suppressMessages(seqdef(ssort, missing=attr(seqdata,'nr'), labels=stlab(seqdata)[matchlab]))
    ##print(alphabet <- c(alphabet, "", attr(ssort.seq,"void"))) ##, attr(ssort.seq,"void")))
    sdss  <- suppressMessages(seqdef(seqdss(ssort.seq)))
    tf <- seqtab(sdss, format="STS", ...)
    t <- attr(tf,"freq")
    ##rownames(t) <- gsub("-\\*","",rownames(t))
    return(t)
}
