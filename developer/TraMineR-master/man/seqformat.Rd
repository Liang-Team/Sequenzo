% Author for TraMineR 2: Pierre-Alexandre Fonta (2016-2017)

\name{seqformat}
\alias{seqformat}
\title{Conversion between sequence representation formats}
\description{
  Convert a sequence data set from one representation format to another.
}
\usage{
seqformat(data, var = NULL, from, to, compress = FALSE, nrep = NULL, tevent,
  stsep = NULL, covar = NULL, SPS.in = list(xfix = "()", sdsep = ","),
  SPS.out = list(xfix = "()", sdsep = ","), id = 1, begin = 2, end = 3,
  status = 4, process = TRUE, pdata = NULL, pvar = NULL, limit = 100,
  overwrite = TRUE, fillblanks = NULL, tmin = NULL, tmax = NULL, missing = "*",
  with.missing = TRUE, right="DEL", compressed, nr)
}
\arguments{
  \item{data}{
    Data frame, matrix, \code{stslist} state sequence object, or character string vector.
    The data to use. (If a tibble, \code{data} is internally converted with \code{as.data.frame}).

    A data frame or a matrix with sequence data in one or more columns when
    \code{from = "STS"} or \code{from = "SPS"}. If sequence data are in a single
    column or in a string vector, they are assumed to be in compressed form (see \code{stsep}).

    A data frame with at least four columns when \code{from = "SPELL"}. Unless specified with the \code{var}, or \code{id} / \code{begin} / \code{end} / \code{status} arguments, the first four columns are assumed to be individual ID, spell
    start time, spell end time, and spell state status.

    A state sequence object when \code{from = "STS"} or \code{from} is not specified.
  }
  \item{var}{
    \code{NULL}, List of Integers or Strings.
    Default: \code{NULL}.
    Indexes or names of the columns containing the sequence information in \code{data}.
    If \code{NULL}, all columns are considered.
  }
  \item{from}{
    String.
    Format of the input sequence data.
    It can be \code{"STS"} (successive states), \code{"SPS"} (successive state-duration spells), or \code{"SPELL"} (vertical id-start-end-state spells). Ignored
    when \code{data} is a \code{stslist} state sequence object.
  }
  \item{to}{
    String.
    Format of the output data.
    It can be \code{"STS"} (successive states), \code{"DSS"} (distinct successive states), \code{"SPS"} (sequences of spells), \code{"SRS"} (shifted replicated sequences), \code{"SPELL"} (vertical spells), or \code{"TSE"} (time stamped events).
  }
  \item{compress}{
    % to STS, DSS, SPS
    Logical.
    Default: \code{FALSE}.
    When \code{to = "STS"}, \code{to = "DSS"}, or \code{to = "SPS"}, should the
    sequences (row vector of states) be concatenated into strings? See
    \code{\link{seqconc}}.
   }
  \item{nrep}{
    % to SRS
    Integer.
    Number of shifted replications when \code{to = "SRS"}.
  }
  \item{tevent}{
    % to TSE
    Matrix.
    The transition-definition matrix when \code{to = "TSE"}.
    It should be of size \eqn{d * d} where \eqn{d} is the number of distinct
    states appearing in the sequences. The cell \eqn{(i,j)} lists the events
    associated with a transition from state \eqn{i} to state \eqn{j}. It can be
    created with \code{\link{seqetm}}.
  }
  \item{stsep}{
    % from STS, SPS
    \code{NULL}, Character.
    Default: \code{NULL}.
    When \code{from = "STS"} or \code{from = "SPS"}, separator token between states 
    in the compressed form (strings).
    If \code{NULL}, \code{\link{seqfcheck}} is called for detecting automatically
    a separator among "-" and ":". Other separators must be explicitly specified.
    See \code{\link{seqdecomp}}.
  }
  \item{covar}{
    % to SRS
    List of Integers or Strings.
    When \code{to = "SRS"}, indexes or names of \code{data} columns to include as
    covariates in the output. Ignored otherwise. Applies only when \code{data} is a data frame with both sequence and covariate data. Must be used in conjunction with \code{var}.
    Covariate values are replicated across the shifted replicated rows.
  }
  \item{SPS.in}{
    % from SPS
    List.
    Default: \code{list(xfix = "()", sdsep = ",")}.
    Specifications for the state-duration couples in the input data when
    \code{from = "SPS"}.
    The first element, \code{xfix}, specifies the prefix/suffix character.
    If a single character, it is used as both prefix and suffix.
    If a two-character string, the first character is used as prefix and the second one as suffix.
    \code{xfix = ""} means no prefix/suffix. The second
    element, \code{sdsep}, specifies the separator token between state and duration.
  }
	\item{SPS.out}{
    % to SPS
    List.
    Default: \code{list(xfix = "()", sdsep = ",")}.
    The specifications for the state-duration couples in the output data when
    \code{to = "SPS"}.
    See \code{SPS.in} above.
	}
	\item{id}{
    % from SPELL
    % to TSE
    \code{NULL}, Integer, String, Vector of Integers or Strings.
    Default: \code{1}.

    When \code{from = "SPELL"}, index or name of the column containing
    the individual IDs in \code{data} (after \code{var} filtering).

    When \code{to = "TSE"}, index or name of the \code{data} column containing
    the individual IDs (after \code{var} filtering), or vector of unique
    individual IDs. If \code{NULL}, indexes of the sequences in the
    input data are used as IDs. If no \code{id} is provided when calling 
    the function and \code{from} is
    not \code{"SPELL"}, \code{id} is set as \code{NULL}.

    When \code{from = "SPELL"} and \code{to = "TSE"}, \code{id} cannot 
    be \code{NULL} and the IDs in the TSE output 
    refer to the IDs in the \code{id} column of  the \code{SPELL} data.
  }
	\item{begin}{
    % from SPELL
    Integer or String.
    Default: \code{2}.
    Index or name of the \code{data} column containing the spell start times
    (after \code{var} filtering) when \code{from = "SPELL"}.
    Start times must be positive integers.
  }
	\item{end}{
	  % from SPELL
    Integer or String.
    Default: \code{3}.
    Index or name of the \code{data} column containing the spell end times 
    (after \code{var} filtering) when \code{from = "SPELL"}.
    End times must be positive integers.
  }
	\item{status}{
		% from SPELL
    Integer or String.
    Default: \code{4}.
    Index or name of the \code{data} column containing the spell statuses
    (after \code{var} filtering) when \code{from = "SPELL"}.
	}
	\item{process}{
	  % from SPELL
    Logical.
    Default: \code{TRUE}.
    When \code{from = "SPELL"}, if \code{TRUE}, create sequences on a process
    time axis, if \code{FALSE}, create sequences on a calendar time axis.
    %If times are ages in \code{data}, only \code{process = FALSE} is currently
    %implemented. %% gr: I do not understand this and have to check.

    This \code{process} argument as well as the associated \code{pdata} and \code{pvar}
    arguments are intended for \code{data} containing spell data with calendar begin
    and end times. When those times are ages, use \code{process = FALSE} with
    \code{pdata=NULL} to use those ages as process times. Option \code{process = TRUE}
    does currently not work for age times.
  }
  \item{pdata}{
    % from SPELL
    % to SPELL
    \code{NULL}, \code{"auto"}, or data frame.
    Default: \code{NULL}. (tibbles are internally converted with \code{as.data.frame}).
    
    To be used only with \code{from = "SPELL"} or \code{to = "SPELL"}.

    If \code{NULL}, start and end times of each spell in the from data are supposed to be ages if
    \code{process = TRUE}, and years if \code{process = FALSE}.

    If \code{"auto"}, ages are computed using 
    the start time of the first spell
    of each individual as her/his birthdate and
    \code{process = TRUE}. For \code{process = FALSE}, \code{"auto"} is equivalent to \code{NULL}.

    A data frame containing the ID and the birth time of the individuals when
    \code{from = "SPELL"} or \code{to = "SPELL"}. Use \code{pvar} to specify
    the column names. The ID is used to match the birth time of each individual
    with the sequence data. The birth time should be integer. It is the start
    time from which the positions on the time axis are computed. It also serves
    to compute \code{tmin} and to guess \code{tmax} when the latter are \code{NULL},
    \code{from = "SPELL"}, and \code{process = FALSE}.
  }
  \item{pvar}{
    % from SPELL
    % to SPELL
    List of Integers or Strings.
    The indexes or names of the columns of the data frame \code{pdata} that
    contain the ID and the birth time of the individuals in that order.
  }
  \item{limit}{
    % from SPELL
    Integer.
    Default: \code{100}.
    The maximum age of age sequences when \code{from = "SPELL"} and \code{process = TRUE}.
    Age sequences will be considered to start at 1 and to end at \code{limit}.
  }
	\item{overwrite}{
    % from SPELL
    Logical.
    Default: \code{TRUE}.
    When \code{from = "SPELL"}, if \code{TRUE}, the most recent episode
    overwrites the older one when they overlap each other, if \code{FALSE},
    the most recent episode starts after the end of the
    previous one.
  }
	\item{fillblanks}{
    % from SPELL
    Character.
    Token used to fill gaps between episodes when \code{from = "SPELL"}.
  }
	\item{tmin}{
    % from SPELL
    \code{NULL} or Integer.
    Default: \code{NULL}.
    When \code{from = "SPELL"} and \code{process = FALSE}, start time of the axis.
    If \code{NULL}, \code{tmin} is set as the earliest spell start time
    (min of \code{begin}) or, when \code{pdata} is a data frame, 
    as the earliest birth time of the individuals.
  }
	\item{tmax}{
    % from SPELL & !process
    \code{NULL} or Integer.
    Default: \code{NULL}.
    When \code{from = "SPELL"} and \code{process = FALSE}, end time of the axis.
    If \code{NULL}, \code{tmax} is set as the latest spell end time 
    (max of \code{end}) or,  when \code{pdata} is a data frame, 
    as the sum of the latest spell end time and the latest
    birth time of the individuals.
  }
  \item{missing}{
    String.
    Default: \code{"*"}.
    Token used for missing states in \code{data}.
    It will be replaced by \code{NA} in the output data. Ignored when \code{data} is a state sequence object
    (see \code{\link{seqdef}}), in which case the attribute \code{nr} is used as missing value token.
  }
  \item{with.missing}{
    % to SPELL
    Logical.
    Default: \code{TRUE}.
    When \code{to = "SPELL"}, should the spells of missing states be included?
  }
  \item{right}{
    % to SPELL
    One of \code{"DEL"} or \code{NA}.
    Default: \code{"DEL"}.
    When \code{to = "SPELL"} and \code{with.missing=TRUE}, set \code{right=NA} to include ending spells of missing states.
  }
  \item{compressed}{
    Deprecated. Use \code{compress} instead.
  }
  \item{nr}{
    Deprecated. Use \code{missing} instead.
  }
}
\details{
  The \code{seqformat} function converts data from one format to
  another. The input data is first converted into STS format and then
  converted into the output format. Depending on input and output formats, some
  information can be lost during the conversion process. The output is a matrix or
  a data frame, NOT a sequence \code{stslist} object. To process, print, and plot
  the sequences with TraMineR functions, you will have to first transform the returned data frame
  into a \code{stslist} state sequence object with \code{\link{seqdef}}.
  See \cite{Gabadinho et al. (2009)} and \cite{Ritschard et al. (2009)} for more
  details on longitudinal data formats and conversion between them.

  When data is in \code{"SPELL"} format (\code{from = "SPELL"}), begin and end times are expected to be positions in the sequences. Therefore, they should be strictly positive integers.
  With \code{process=TRUE}, the outcome sequences will be aligned on ages (process duration since birth), while with \code{process=FALSE} they will be aligned on dates (position on the calendar time). If \code{process=TRUE}, values in the \code{begin} and \code{end} columns of \code{data} are assumed to be ages when \code{pdata} is \code{NULL} and integer dates otherwise. If \code{process=FALSE}, begin and end values are assumed to be integer dates when \code{pdata} is \code{NULL} and ages otherwise.
  
  To convert from person-period data use \code{from = "SPELL"} and set both \code{begin} and \code{end} as the index or name of the time (period) column. Alternatively, use the \code{\link{reshape}} command of \code{stats}, which is more efficient.
}
\value{
  A data frame for \code{SRS}, \code{TSE}, and \code{SPELL} outcomes, otherwise a matrix.

  When \code{from="SPELL"}, outcome has an attribute \code{issues} with indexes of sequences with issues (truncated sequences, missing start time, spells before birth year, ...)
}
\references{
  Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2009). Mining
  Sequence Data in \code{R} with the \code{TraMineR} package: A user's guide.
  Department of Econometrics and Laboratory of Demography, University of Geneva.

  Ritschard, G., A. Gabadinho, M. Studer and N. S. Müller. Converting between
  various sequence representations. in Ras, Z. & Dardzinska, A. (eds.)
  \emph{Advances in Data Management}, Springer, 2009, 223, 155-175.
}
\author{
  Gilbert Ritschard, Alexis Gabadinho, Pierre-Alexandre Fonta, Nicolas S. Müller, Matthias Studer
}
\seealso{\code{\link{seqdef}}, \code{\link{reshape}}}
\examples{
## ========================================
## Examples with raw STS sequences as input
## ========================================

## Loading a data frame with sequence data in the columns 13 to 24
data(actcal)

## Converting to SPS format
actcal.SPS.A <- seqformat(actcal, 13:24, from = "STS", to = "SPS")
head(actcal.SPS.A)

## Converting to compressed SPS format with no
## prefix/suffix and with "/" as state/duration separator
actcal.SPS.B <- seqformat(actcal, 13:24, from = "STS", to = "SPS",
  compress = TRUE, SPS.out = list(xfix = "", sdsep = "/"))
head(actcal.SPS.B)

## Converting to compressed DSS format
actcal.DSS <- seqformat(actcal, 13:24, from = "STS", to = "DSS",
  compress = TRUE)
head(actcal.DSS)


## ==============================================
## Examples with a state sequence object as input
## ==============================================

## Loading a data frame with sequence data in the columns 10 to 25
data(biofam)

## Limiting the number of considered cases to the first 20
biofam <- biofam[1:20, ]

## Creating a state sequence object
biofam.labs <- c("Parent", "Left", "Married", "Left/Married",
  "Child", "Left/Child", "Left/Married/Child", "Divorced")
biofam.short.labs <- c("P", "L", "M", "LM", "C", "LC", "LMC", "D")
biofam.seq <- seqdef(biofam, 10:25, alphabet = 0:7,
  states = biofam.short.labs, labels = biofam.labs)

## Converting to SPELL format
bf.spell <- seqformat(biofam.seq, from = "STS", to = "SPELL",
  pdata = biofam, pvar = c("idhous", "birthyr"))
head(bf.spell)

## Converting to shifted replicated sequences (SRS)
bf.srs <- seqformat(biofam, var=10:25, from="STS", to="SRS", 
                    covar=c("sex","plingu02"))
tail(bf.srs)


## ======================================
## Examples with SPELL sequences as input
## ======================================

## Loading two data frames: bfspell20 and bfpdata20
## bfspell20 contains the first 20 biofam sequences in SPELL format
## bfpdata20 contains the IDs and the years at which the
## considered individuals were aged 15
data(bfspell)

## Converting to STS format with alignement on calendar years
bf.sts.y <- seqformat(bfspell20, from = "SPELL", to = "STS",
  id = "id", begin = "begin", end = "end", status = "states",
  process = FALSE)
head(bf.sts.y)

## Converting to STS format with alignement on ages
bf.sts.a <- seqformat(bfspell20, from = "SPELL", to = "STS",
  id = "id", begin = "begin", end = "end", status = "states",
  process = TRUE, pdata = bfpdata20, pvar = c("id", "when15"),
  limit = 16)
names(bf.sts.a) <- paste0("a", 15:30)
head(bf.sts.a)


## ==================================
## Examples for TSE and SPELL output
## in presence of missing values
## ==================================

data(ex1) ## STS data with missing values
## creating the state sequence object with by default
## the end missings coded as void ('%')
sqex1 <- seqdef(ex1[,1:13])
as.matrix(sqex1)

## Creating state-event transition matrices
ttrans <- seqetm(sqex1, method='transition')
tstate <- seqetm(sqex1, method='state')

## Converting into time stamped events
seqformat(sqex1, from = "STS", to = "TSE", tevent = ttrans)
seqformat(sqex1, from = "STS", to = "TSE", tevent = tstate)

## Converting into vertical spell data
seqformat(sqex1, from = "STS", to = "SPELL", with.missing=TRUE)
seqformat(sqex1, from = "STS", to = "SPELL", with.missing=TRUE, right=NA)
seqformat(sqex1, from = "STS", to = "SPELL", with.missing=FALSE)
}
\keyword{Data handling}
