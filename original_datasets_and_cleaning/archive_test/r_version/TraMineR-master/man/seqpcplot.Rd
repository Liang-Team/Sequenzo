%% -*- TeX -*-
\name{seqpcplot}
\alias{seqpcplot}
\alias{seqpcfilter}
\alias{plot.eseq}
\alias{plot.seqelist}
\title{Parallel coordinate plot for sequence data}
\description{
  A decorated parallel coordinate plot to render the order of the
  successive elements in sequences. The sequences are displayed as
  jittered frequency-weighted parallel lines.
  The plot is also embedded as the \code{type="pc"} option of the
  \code{\link{seqplot}} function and serves as plot
  method for \code{eseq} and \code{seqelist} objects.
}


\usage{
seqpcplot(seqdata, group = NULL, weights = NULL, cex = 1, lwd = 1/4,
  cpal = NULL, grid.scale = 1/5, ltype = "unique",
  embedding = "most-frequent", lorder = NULL , lcourse = "upwards",
  filter = NULL, hide.col = "grey80", alphabet = NULL,
  missing = "auto", order.align = "first", main = "auto", xlab = NULL,
  ylab = NULL, xaxis = TRUE, yaxis = TRUE, axes = "all", xtlab = NULL,
  cex.lab = 1, rows = NA, cols = NA, plot = TRUE, seed = NULL,
  weighted = TRUE, with.missing = TRUE,
  title, cex.plot, ...)

seqpcfilter(method = c("minfreq", "cumfreq", "linear"), level = 0.05)

}
\arguments{
  \item{seqdata}{The sequence data. Either an event sequence
    object of class \code{seqelist} (see
    \code{\link{seqecreate}}) or a state sequence object of class
    \code{stslist} (see \code{\link{seqdef}}).}

  \item{group}{a vector (numeric or factor) of group memberships
    of length equal the number of
    sequences. When specified, one plot is generated for each
    different membership value.}

  \item{weights}{a numeric vector of weights of length equal the number
    of sequences. When \code{NULL}, the \code{weights} are taken from
    the \code{seqdata} object.}

  \item{cex}{Plotting text and symbols magnification. See \code{\link{par}}.}

  \item{lwd}{expansion factor for line widths. The expansion is
    relative to the size of the squared symbols.}

  \item{cpal}{color palette vector for line coloring.}

  \item{grid.scale}{Expansion factor for the translation zones.}

  \item{ltype}{the type of sequence that is drawn. Either \code{"unique"}
    to render unique patterns or \code{"non-embeddable"} to render
    non-embeddable sequences.}

  \item{embedding}{The method for embedding sequences embeddable in
    multiple non-embeddable sequences. Either \code{"most-frequent"}
    (default) or \code{"uniformly"}. Relevant only with \code{ltype =
      "non-embeddable"}.}

  \item{lorder}{line ordering. Either \code{"background"} or
  \code{"foreground"}.}

  \item{lcourse}{Method to connect simultaneous elements with the
    preceding and following ones. Either \code{"upwards"} (default) or
    \code{"downwards"}.}

  \item{filter}{list of line coloring options. See details.}

  \item{hide.col}{Color for sequences filtered-out by the
    \code{filter} specification.}

  \item{alphabet}{a vector of response levels in the order they should
    appear on the y-axis. This argument is solely relevant for
    \code{seqelist} objects.}

  \item{missing}{character. Whether and how missing values should be
    displayed. Available are \code{"auto"}, \code{"show"} and
    \code{"hide"}. If \code{"auto"}, the plot will show missings only if
    present. \code{"hide"} will fade out missings and \code{"show"} will
    always show missings. If \code{with.missing=FALSE},
    \code{missing} is turned into \code{"hide"}. If \code{with.missing=TRUE}
    and \code{missing="hide"}, \code{missing} is turned into \code{"auto"}.}

  \item{order.align}{Aligning method. For aligning on order positions use either \code{"first"} (default) or
  \code{"last"}. Option \code{"first"} numbers the positions from the beginning
  while \code{"last"} numbers them from the end. With \code{order.align = "time"}, the elements in the
  sequences are aligned on their rounded timestamps.}

  \item{main}{title for the graphic. Default \code{"auto"} prints default titles. Set as \code{NULL} to
  suppress the title.}
  \item{xlab}{label for the x-axis}
  \item{ylab}{label for the y-axis}
  \item{xaxis}{logical: Should x-axis be plotted?}
  \item{yaxis}{logical: Should y-axis be plotted?}

  \item{axes}{if set as \code{"all"} (default value) x-axes are drawn
    for each plot in the graphic. If set as \code{"bottom"} and
    \code{group} is used, axes are drawn only under the plots at
    the bottom of the graphic area. If \code{FALSE}, no x-axis is drawn.}

  \item{xtlab}{labels for the x-axis ticks.}

  \item{cex.lab}{x and y labels magnification. See \code{\link{par}}.}

  \item{rows,cols}{integers. Number of rows and columns of the plot panel.}

  \item{plot}{logical. Should the plot be displayed? Set as \code{FALSE} to retrieve the \code{seqpcplot}
  object without plotting it.}

  \item{seed}{integer. Start seed value.}

  \item{weighted}{logical. Should weights be accounted for? Default is \code{TRUE}.}

  \item{with.missing}{logical. Should we care about possible missings? Default is \code{TRUE}. See also the
  \code{missing} argument.}

  \item{method}{character string. Defines the filtering
     function. Available are \code{"minfreq"}, \code{"cumfreq"} and
     \code{"linear"}.}

  \item{level}{numeric scalar between 0 and 1. The frequency threshold
    for the filtering methods \code{"minfreq"} and \code{"cumfreq"}.}

  \item{title}{Deprecated. Use \code{main} instead.}
  \item{cex.plot}{Deprecated. Use \code{cex.lab} instead.}
  \item{...}{arguments to be passed to other methods, such as graphical
    parameters (see \code{\link{par}}).}
}
\details{
  For plots by groups specified with the \code{group} argument, plotted
  line widths and point sizes reflect relative frequencies within
  group.

  The \code{filter} argument serves to specify filters to gray less
  interesting patterns. The filtered-out patterns are displayed in the
  \code{hide.col} color. The \code{filter} argument expects a list with
  at least elements \code{type} and \code{value}. The following types
  are implemented:

  Type \code{"sequence"}: colors a specific pattern, for example assign\cr
  \code{filter = list(type = "sequence", value = "(Leaving
  Home,Union)-(Child)")}.

  Type \code{"subsequence"}: colors patterns which include a specific
  subsequence, for example\cr \code{filter = list(type =
  "subsequence", value = "(Child)-(Marriage)")} .

  Type \code{"value"}: gradually colors the patterns according to the
  numeric vector (of length equal to the number of sequences) provided as
  \code{"value"} element in the list. You can give something like
  \code{filter = list(type = "value", value = c(0.2, 1, ...))} or
  provide the distances to the medoid as \code{value} vector for
  example.

  Type \code{"function"}: colors the patterns depending on the values
  returned by a [0,1] valued function of the frequency x of the
  pattern. Three native functions can be used: \code{"minfreq"},
  \code{"cumfreq"} and \code{"linear"}. Use
  \code{filter = list(type = "function", value = "minfreq", level = 0.05)}
  to color patterns with a
  support of at least 5\% (within group). Use
  \cr\code{filter = list(type = "function", value = "cumfreq", level = 0.5)}
  to highlight the 50\% most
  frequent patterns (within group). Or, use
  \code{filter = list(type="function", value="linear")}
  to use a linear gradient for the
  color intensity (the most frequent trajectory gets
  100\% intensity). Other user-specified functions can be provided by
  giving something like
  \cr\code{filter = list(type="function", value=function(x, arg1, arg2) {return(x/max(x) * arg1/arg2)},}
  \code{arg1 = 1, arg2 = 1)}. This latter function adjusts gradually the color intensity
  of patterns according to the frequency of the pattern.

  The function \code{seqpcfilter} is a convenience function for type
  \code{"function"}. The three examples above can be imitated by
  \code{seqpcfilter("minfreq", 0.05)},
  \code{seqpcfilter("cumfreq", 0.5)} and \code{seqpcfilter("linear")}.

  If a numeric scalar is assigned to \code{filter}, the \code{"minfreq"}
  filter is used.
}

\value{
  An object of \code{class} \code{"seqpcplot"}
  with various information necessary for constructing the plot,
  e.g. coordinates. There is a \code{summary} method for such
  objects.
}


\seealso{ \code{\link{seqplot}}, \code{\link{seqdef}}, \code{\link{seqecreate}}}

\references{
  Bürgin, R. and G. Ritschard (2014), A decorated parallel coordinate plot for categorical longitudinal
  data, \emph{The American Statistician} 68(2), 98-103.
}

\examples{

## ================
## plot biofam data
## ================

data(biofam)
lab <- c("Parent","Left","Married","Left+Marr","Child","Left+Child",
         "Left+Marr+Child","Divorced")

## plot state sequences in STS representation
## ==========================================

## creating the weighted state sequence object.
biofam.seq <- seqdef(data = biofam[,10:25], labels = lab,
                     weights = biofam$wp00tbgs)

## select the first 20 weighted sequences (sum of weights = 18)
biofam.seq <- biofam.seq[1:20, ]

par(mar=c(4,8,2,2))
seqpcplot(seqdata = biofam.seq, order.align = "time")

## .. or
seqplot(seqdata = biofam.seq, type = "pc", order.align = "time")

## Distinct successive states (DSS)
## ==========================================

seqplot(seqdata = biofam.seq, type = "pc", order.align = "first")

## .. or (equivalently)

biofam.DSS <- seqdss(seqdata = biofam.seq) # prepare format
seqpcplot(seqdata = biofam.DSS)

## plot event sequences
## ====================

biofam.eseq <- seqecreate(biofam.seq, tevent = "state") # prepare data

## plot the time in the x-axis
seqpcplot(seqdata = biofam.eseq, order.align = "time", alphabet = lab)

## ordering of events
seqpcplot(seqdata = biofam.eseq, order.align = "first", alphabet = lab)

## ... or
plot(biofam.eseq, order.align = "first", alphabet = lab)

## additional arguments
## ====================

## non-embeddable sequences
seqpcplot(seqdata = biofam.eseq, ltype = "non-embeddable",
          order.align = "first", alphabet = lab)

## align on last event
par(mar=c(4,8,2,2))
seqpcplot(seqdata = biofam.eseq, order.align = "last", alphabet = lab)

## use group variables
seqpcplot(seqdata = biofam.eseq, group = biofam$sex[1:20],
          order.align = "first", alphabet = lab)

## color patterns (Parent)-(Married) and (Parent)-(Left+Marr+Child)
par(mfrow = c(1, 1))
seqpcplot(seqdata = biofam.eseq,
          filter = list(type = "sequence",
                          value=c("(Parent)-(Married)",
                                  "(Parent)-(Left+Marr+Child)")),
          alphabet = lab, order.align = "first")

## color subsequence pattern (Parent)-(Left)
seqpcplot(seqdata = biofam.eseq,
          filter = list(type = "subsequence",
                          value = "(Parent)-(Left)"),
          alphabet = lab, order.align = "first")

## color sequences over 10% (within group) (function method)
seqpcplot(seqdata = biofam.eseq,
          filter = list(type = "function",
                        value = "minfreq",
                        level = 0.1),
          alphabet = lab, order.align = "first", seed = 1)

## .. same result using the convenience functions
seqpcplot(seqdata = biofam.eseq,
          filter = 0.1,
          alphabet = lab, order.align = "first", seed = 1)

seqpcplot(seqdata = biofam.eseq,
          filter = seqpcfilter("minfreq", 0.1),
          alphabet = lab, order.align = "first", seed = 1)

## highlight the 50% most frequent sequences
seqpcplot(seqdata = biofam.eseq,
          filter = list(type = "function",
                          value = "cumfreq",
                          level = 0.5),
          alphabet = lab, order.align = "first", seed = 2)

## .. same result using the convenience functions
seqpcplot(seqdata = biofam.eseq,
          filter = seqpcfilter("cumfreq", 0.5),
          alphabet = lab, order.align = "first", seed = 2)

## linear gradient
seqpcplot(seqdata = biofam.eseq,
          filter = list(type = "function",
                          value = "linear"),
          alphabet = lab, order.align = "first", seed = 2)

seqpcplot(seqdata = biofam.eseq,
          filter = seqpcfilter("linear"),
          alphabet = lab, order.align = "first", seed = 1)



}

\author{Reto Bürgin (with Gilbert Ritschard for the help page)}

\keyword{Plot}
\keyword{Event sequences}
