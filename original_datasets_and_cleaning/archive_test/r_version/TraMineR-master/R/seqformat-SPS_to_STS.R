# Should only be used through seqformat()

SPS_to_STS <- function(seqdata, spsformat, missing = "*") {
  nseq <- nrow(seqdata)
  trans <- matrix("", nrow = nseq, ncol = 1)
  xfix <- spsformat$xfix
  if (!identical(xfix, ""))
    xfix <- paste0("[", xfix, "]")
  sdsep <- spsformat$sdsep
  for (i in 1:nseq) {
    tmpseq <- na.omit(seqdata[i, ])
    if (length(tmpseq)>0) {
      for (s in 1:length(tmpseq)) {
        sps <- strsplit(gsub(xfix, "", tmpseq[s]), split = sdsep)[[1]]
        seq <- sps[1]
        if (seq %in% missing)
          seq <- NA
        dur <- as.integer(sps[2])
        trans[i] <-
          if (s == 1)
            paste(trans[i], seq, sep = "")
          else
            paste(trans[i], seq, sep = "-")
        if (dur > 1)
          for (r in 2:dur)
            trans[i] <- paste(trans[i], "-", seq, sep = "")
      }
    }
  }
  sts <- seqdecomp(trans)
  return(sts)
}
