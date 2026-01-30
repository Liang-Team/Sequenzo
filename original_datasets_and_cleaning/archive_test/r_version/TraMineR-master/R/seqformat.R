# Author for TraMineR 2: Pierre-Alexandre Fonta (2016-2017)

seqformat <- function(data, var = NULL, from, to, compress = FALSE, nrep = NULL,
  tevent, stsep = NULL, covar = NULL, SPS.in = list(xfix = "()", sdsep = ","),
  SPS.out = list(xfix = "()", sdsep = ","), id = 1, begin = 2, end = 3,
  status = 4, process = TRUE, pdata = NULL, pvar = NULL, limit = 100,
  overwrite = TRUE, fillblanks = NULL, tmin = NULL, tmax = NULL, missing = "*",
  with.missing = TRUE, right="DEL", compressed, nr) {

  TraMineR.check.depr.args(alist(compress = compressed, missing = nr))

  ## tibble converted to data frame
  if (inherits(data, "tbl_df")) data <- as.data.frame(data)
  if (inherits(pdata, "tbl_df")) pdata <- as.data.frame(pdata)
  is.stslist <- if (inherits(data, "stslist")) TRUE else FALSE

  #### Check arguments with deprecated values ####

  if (is.strings(data) & !is.matrix(data)) {
    data <- as.matrix(data)
    #msg.warn("'data' as a string is deprecated, use as.matrix() to convert it")
  }

  if (!is.null(covar) && to != "SRS"){
    msg.warn('covar ignored when to is not "SRS"')
    covar <- NULL
  }

  #### Check for arguments that need to be defined ####

  # from
  if (missing(from)) {
    if (is.stslist) {
      from <- "STS"
      msg("'from' set to 'STS' as 'data' is a sequence object")
    } else {
      msg.stop.miss("from")
    }
  }

  # to
  if (missing(to))
    msg.stop.miss("to")

  #### Check argument types ####

  # from
  # Add new input format names here
  formats.from <- c("STS", "SPS", "SPELL")
  if (! from %in% formats.from)
    msg.stop.in("from", formats.from)

  # to
  # Add new output format names here
  formats.to <- c("STS", "DSS", "SPS", "SRS", "TSE", "SPELL")
  if (! to %in% formats.to)
    msg.stop.in("to", formats.to)

  # data
  if (from == "STS" && !any(is.stslist, is.data.frame(data), is.matrix(data)))
    msg.stop("'data' must be a state sequence object, a data frame, or a matrix")

  if (from == "SPS" && (is.stslist || !any(is.data.frame(data), is.matrix(data))))
    msg.stop("'data' must be a data frame or a matrix")

  if (from == "SPELL" && (is.stslist || !all(is.data.frame(data), ncol(data) >= 3)))
    msg.stop("'data' must be a data frame with at least three columns")

  # var
  if (!is.null(var))
    checkindexes(var,"var")

  # missing
  #missing <- as.character(missing)
  if (!is.a.string(missing[1]))
    msg.stop.na("missing")

  #### Check format specific arguments ####

  # from STS or SPS
  if (from %in% c("STS", "SPS") && !is.null(stsep) && !is.a.character(stsep))
    msg.stop("'stsep' must be a character")

  # to SRS
  if (!is.null(covar) & to == "SRS")
    checkindexes(covar,"covar")

  # to TSE
  if (to == "TSE" && missing(tevent))
    msg.stop.miss("tevent")

  # from SPELL
  if (from == "SPELL") {
    checkindex(id, "id")
    checkindex(begin, "begin")
    checkindex(end, "end")
    checkindex(status, "status")
    if (!is.a.number(limit))
      msg.stop("'limit' must be a number")
    if (!is.null(fillblanks) && !is.a.character(fillblanks))
      msg.stop("'fillblanks' must be a character")
  }

  # from / to SPELL
  if (from == "SPELL" || to == "SPELL") {
    if (!is.null(pvar)) {
      checkindexes(pvar, "pvar")
      if (!is.data.frame(pdata))
        msg.warn0("'pvar' ignored because 'pdata' is not a data frame")
    } else {
      if (is.data.frame(pdata))
        msg.stop("'pvar' required when 'pdata' is a data frame")
    }
  }

  #### Compute format specific values ####

  # STS
  if (is.stslist) {
    void <- attr(data, "void")
    missing <- attr(data, "nr")
    msg.warn0("'missing' set as \"c(\'",missing,"\',\'",void,"\')\", the 'nr' and 'void' code from the 'data' state sequence object")
    missing <- c(missing, void)
  }

  # TSE
  if (to == "TSE") {
    if (from != "SPELL") {
      if (missing(id)) {
        uids <- NULL
        msg.warn("'id' set to NULL as it is not specified (backward compatibility with TraMineR 1.8)")
      } else if (is.null(id)) {
        uids <- NULL
      } else if (!is.null(id)) {
        lid <- length(id)
        if (lid == 1) {
          uids <- unique(data[, id])
        } else if (lid > 1) {
          if (length(unique(id)) != lid)
            msg.stop("'id' must contain unique IDs")
          else
            uids <- id
        } else {
          msg.stop.na("id")
        }
      }
    }
    if (from != "SPELL" && is.null(uids))
      msg.warn("replacing original IDs in the output by the sequence indexes")
    else
      msg("using original IDs in the output")
  }

  #### Convert input format into internal STS format ####

  # Extract sequence data from the dataset
  seqdata <- if (!is.null(var)) subset(data, , var) else data

  ncols <- ncol(seqdata)
  if (ncols == 0)
    msg.stop("'data' must contain at least one column after 'var' filtering")

  if (from != "SPELL") {
    # Check if the input data are compressed
    # Add new compressed formats here
    if (from %in% c("STS", "SPS") && ncols == 1)
      is.compressed <- TRUE
    else
      is.compressed <- FALSE

    # Convert input data to a matrix
    # No need to reformat columns here as sequence data are expected to be characters
    mseqdata <- as.matrix(seqdata)

    # Check if the separator of compressed data is '-' or ':'
    if (is.compressed && is.null(stsep)) {
      stsep.auto <- seqfcheck(mseqdata)
      if (! stsep.auto %in% c("-", ":") && max(nchar(seqdata)>1)) {
        msg.stop("'stsep' must be specified as it is neither '-' nor ':'")
      } else {
        stsep <- stsep.auto
        msg0("setting 'stsep' as \"", stsep, "\" (autodetected)")
      }
    }

    # Decompress compressed data
    # Replace 'missing' code by NA
    if (is.compressed)
      mseqdata <- seqdecomp(mseqdata, sep = stsep, miss = as.character(missing))
    # Note: seqdecomp() inserts neither 'nr' nor 'void' codes
    # TODO seqdecomp() uses NA as 'void' code
  }

  # Replace 'missing' code by NA
  # For from SPS, see SPS_to_STS()
  if (from != "SPS") {
    if (from != "SPELL") {
      if (to != "TSE") {
        mseqdata[mseqdata %in% missing] <- NA
      }
    } else {
      seqdata[, status][seqdata[, status] %in% missing] <- NA
    }
  }

  # Call input format specific processing functions
  # Add new input format processing functions here
  msts <- switch(from,
    STS = mseqdata,
    SPS = SPS_to_STS(mseqdata, SPS.in, missing),
    SPELL = SPELL_to_STS(seqdata, id, begin, end, status, process, pdata, pvar,
      limit, overwrite, fillblanks, tmin, tmax))
  # Note: In mseqdata, 'nr' and 'void' codes are unchanged
  # Note: SPS_to_STS() inserts neither 'nr' nor 'void' codes
  # Note: SPELL_to_STS() inserts neither 'nr' nor 'void' codes
  # TODO SPELL_to_STS() uses NA as 'void' code

  nseqs.in <- nrow(msts)

  # Add new long formats here
  if (from %in% c("SPELL"))
    msg("converting", from, "data into", nseqs.in, "STS sequences (internal format)")

  #### Convert internal STS format into output format ####

  # to SPELL
  if (to == "SPELL")
    ssts <- suppressMessages(seqdef(msts, missing = NA, right = right))
  # Note: seqdef() should use the default 'nr' code because of [1]
  # Note: seqdef() replaces the 'missing' code by the 'nr' code
  # Note: seqdef() inserts 'void' codes

  # from SPELL with to TSE
  if (from == "SPELL" && to == "TSE")
    uids <- unique(rownames(msts))
  # Note: SPELL_to_STS() stores IDs in the row names

  # Call output format specific processing functions
  # Output is a data frame for SRS, TSE, and SPELL, otherwise a matrix
  # Add new output format processing functions here
  converted <- switch(to,
    STS = msts,
    DSS = suppressMessages(STS_to_DSS(msts, missing = NA)),
  #  SPS = suppressMessages(STS_to_SPS(msts, SPS.out, missing = NA)),
    SPS = suppressMessages(STS_to_SPS(msts, SPS.out, right=right, missing = NA)),
    SRS = STS_to_SRS(msts, nrep),
    TSE = STS_to_TSE(msts, uids, tevent),
    SPELL = STS_to_SPELL(ssts, pdata, pvar, with.missing))
  # Note: In msts, 'nr' and 'void' codes are unchanged
  # Note: STS_to_DSS() and STS_to_SPS(): error if 'nr = NA', suppressMessages() for seqprep()
  # Note: STS_to_DSS() and STS_to_SPS() replace 'missing' code by 'nr' code and insert 'void' codes
  # Note: STS_to_SRS() doesn't insert 'nr' nor 'void' codes
  # Note: STS_to_TSE() doesn't insert 'nr' nor 'void' codes
  # Note: [1] STS_to_SPELL() replaces 'nr' code by 'void' code if with.missing = FALSE
  # and 'nr' code is included as a factor level otherwise

  #### Post-processing ####

  # to SRS
  if (to == "SRS" && !is.null(covar)) {
    # Use subset() to keep the column name when there is olny one 'covar' column
    converted <- merge(converted, data.frame(id = seq(1:nseqs.in), subset(data, , covar)))
    msg("adding covariates", paste(covar, collapse = ", "))
  }

  # Modify existing or add new final message depending on the output format here
  # Must be before compressing for a meaningful succession of information messages
  nseqs.out <- nrow(converted)
  if (! to %in% c("STS", "SPELL"))
    msg("converting STS sequences to", nseqs.out, to, "sequences")
  else if (to == "SPELL")
    msg("converting STS sequences to", nseqs.out, "spells")

  # TODO Harmonize output type.
  # TODO Experiments, Pierre-Alexandre Fonta (2016-2017).
  # TODO Commented because too much projects use the unharmonized output types.
  # Convert output data frame to a matrix
  # Reformat colums to avoid insertion of white spaces
  # Don't encode NA to avoid having them as characters in the compressed strings
  # if (is.data.frame(converted)) {
  #   converted <- sapply(converted, format, trim = TRUE, justify = "none", na.encode = FALSE)
  #   converted <- as.matrix(converted)
  # } else if (!is.matrix(converted)) {
  #   msg.stop("output value must be a data frame or a matrix")
  # }

  # TODO Harmonize 'nr' / 'missing' code handling.
  # TODO Experiments, Pierre-Alexandre Fonta (2016-2017).
  # TODO Commented because too much implicit logic uses the unharmonized missing
  #      values handling.
  # Replace 'nr' code by NA
  # Must be done before potential compression with seqconc() to avoid having 'nr'
  # in the strings.
  # 'nr' is obtained from 'data' if it is a state sequence object, otherwise it
  # is assumed to be the default 'nr', "*" (see seqprep() and seqdef()).
  # converted[converted == missing] <- NA

  # to STS, DSS, SPS
  # Compression
  # Add new compressible formats here
  if (isTRUE(compress)) {
    if (to %in% c("STS", "DSS", "SPS")) {
      converted <- seqconc(converted)
      # Note: seqconc() returns a matrix
      # Note: seqconc() data must not have factors columns because of paste()
      # Note: [2] seqconc() doesn't include 'void' elements in the string
      msg("compressing", to, "sequences")
    } else {
      msg.stop(to, "is not a compressible format")
    }
  }

  # TODO Harmonize 'void' code handling (including in the input data).
  # TODO Experiments, Pierre-Alexandre Fonta (2016-2017).
  # TODO Commented because too much implicit logic uses the unharmonized 'void'
  #      code handling.
  # Replace 'void' code by NA
  # Must be done after potential compression with seqconc() because of [2].
  # 'void' is obtained from 'data' if it is a state sequence object, otherwise
  # it is assumed to be the default, "%" (see seqprep() and seqdef()).
  # if (!is.stslist)
  #   void <- "%"
  # converted[converted == void] <- NA

  return(converted)
}
