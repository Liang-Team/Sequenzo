\name{HSPELL_to_STS}
\alias{HSPELL_to_STS}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Data conversion from Horizontal Spell to STS.
}
\description{
Convert data from Horizontal Spell to STS.
}
\usage{
HSPELL_to_STS(seqdata, begin, end, status = NULL,
    fixed.status = NULL, pvar = NULL, overwrite = TRUE,
    fillblanks = NULL, tmin = NULL, tmax = NULL, id = NULL,
    endObs = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqdata}{
a data frame or matrix containing sequence data.
}
  \item{begin}{
Vector containing the columns (name or number) with the beginning position of each spell.
}
  \item{end}{
Vector containing the columns (name or number) with the end position of each spell.
}
  \item{status}{
Vector containing the columns (name or number) with the status of each spell.
}
  \item{fixed.status}{ Default status (for period not covered by any spell.)
}
  \item{pvar}{ names or numbers of the column containing the 'birth' time.
}
  \item{overwrite}{Should the most recent episode overwrite the older one when they overlap? If \code{FALSE}, the most recent episode starts from the end of the previous one.
}
  \item{fillblanks}{If not \code{NULL}, character used for filling gaps between episodes.
}
  \item{tmin}{
If sequences are to be defined on a calendar time axis, it defines the starting time of the axis. If set as NULL, the start time is set as the minimum of the 'begin' column in the data.}
  \item{tmax}{
If year sequences are wanted, defines the ending year of the sequences. If set to NULL, it is guessed from the data (not so accurately!).
}
  \item{id}{
column containing the identification numbers for the sequences.
}
  \item{endObs}{An optional end of observation date. Usefull for retrospective survey.}
}
\details{
Hortizontal spell data format has the following caracteristics:
    - One row per individual
    - Each spell is specified with three consecutive variables: a begin date, an end date, and the status.
    - For unused spells, begin and end values should be set as \code{NA}.
}
\value{
A \code{data.frame} with the sequence in STS format.
}
\author{
Matthias Studer
}

\seealso{
See Also \code{\link[TraMineR]{seqformat}}.
}
\note{This function is a pre-release and further testing is still needed, please report any problems.}
\examples{
hspell <- data.frame(begin1=rep(1, 5), end1=c(2:5, NA), status1=1:5,
                     begin2=c(3:6, NA), end2=rep(NA, 5), status2=5:1)
sts <- HSPELL_to_STS(hspell, begin=c("begin1", "begin2"), end=c("end1", "end2"),
                     status=c("status1", "status2"))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{data format}
