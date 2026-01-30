\name{plot.stslist}
\alias{plot.stslist}
\title{Plot method for state sequence objects}
\description{
This is the plot method for state sequence objects of class \code{stslist} created by the \code{seqdef} function. It produces a sequence index plot.
}
\usage{
\method{plot}{stslist}(x, idxs = NULL, weighted = TRUE, sortv = NULL,
  cpal = NULL, missing.color = NULL, ylab = NULL,
  yaxis = TRUE, xaxis = TRUE, ytlab = NULL, las = par("las"),
  xtlab = NULL, xtstep = NULL, tick.last = NULL, cex.axis = par("cex.axis"),
  tlim, cex.plot, ylas, ...)
}
\arguments{
	\item{x}{A state sequence object created with the \code{\link{seqdef}} function.}

	\item{idxs}{Indexes of the sequences to be plotted (default value is \code{1:10}), for instance \code{20:50} to plot sequences 20 to 50, \code{c(2,8,12,25)} to plot sequences 2,8,12 and 25 in \code{seqdata}. If set to \code{0}, all sequences in \code{seqdata} are plotted.}

      \item{weighted}{Logical: Should the bar representing each sequence be proportional to its weight? Ignored when no weights are assigned to sequences (see \code{\link{seqdef}}.)}

	\item{sortv}{A sorting variable or a sort method (one of \code{"from.start"} or \code{"from.end"}). See details.}

	\item{cpal}{Color palette for the states. A vector of colors of length equal to the number of states in the alphabet. If \code{NULL} (default), the \code{cpal} attribute of the \code{seqdata} sequence object is used (see \code{\link{seqdef}}).}

	\item{missing.color}{Color for representing missing values inside the sequences. If \code{NULL} (default) the color is taken from the "missing.color" attribute of the \code{x} sequence object.}

	\item{ylab}{String. Optional label of the y-axis. If set to \code{NA}, no label is drawn.}

	\item{yaxis}{Logical. Should the y-axis be plotted. When set as \code{TRUE}, sequence indexes are displayed.}

	\item{xaxis}{Logical. Should the x-axis (time) be plotted? Default is \code{TRUE}.}

	\item{ytlab}{Character string or vector of length equal to the number of  sequences. Labels of the plotted sequences to be used as tick labels on the y-axis. Default is the indexes of the sequences as defined by the \code{idxs} argument. Can be set to \code{"id"} for displaying the row names (id) of the sequences instead of their indexes; row names can be assigned to the sequence object with the  \code{id} argument of the \code{\link{seqdef}} function or afterwards with \code{\link{rownames}}. Otherwise \code{ytlab} can be set to a vector of length equal to the number of  sequences to be plotted.}

  \item{las}{numeric in \{0,1,2,3\}; the orientation of tick labels.
  \cr
0: always parallel to the axis (default),
    \cr
1: always horizontal,
    \cr
2: always perpendicular to the axis,
    \cr
3: always vertical.
}

  \item{xtlab}{optional labels for the x-axis tick labels. If unspecified, the column names of the \code{seqdata} sequence object are used (see \code{\link{seqdef}}).}

  \item{xtstep}{optional interval at which tick marks and labels of the x-axis are displayed. For example, with \code{xtstep=3} a tick mark is drawn at position 1, 4, 7, etc... The display of the corresponding labels depends on the available space and is dealt with automatically. If unspecified, the \code{xtstep} attribute of the \code{x} object is used.}

  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, the \code{tick.last} attribute of the \code{x} object is used.}

  \item{cex.axis}{Axis annotation magnification. See \code{\link{par}}.}

  \item{tlim}{Deprecated. Use \code{idxs} instead.}
  \item{cex.plot}{Deprecated. Use \code{cex.axis} instead.}
  \item{ylas}{Deprecated. Use \code{las} instead.}

 \item{...}{further graphical parameters (see \code{\link{par}}) and \code{\link{barplot}} arguments.}
}

\details{
This is the default plot method for state sequence objects (produced by the \code{\link{seqdef}} function), i.e., for objects of class \emph{stslist}. It produces a sequence index plot, where individual sequences are rendered with stacked bars depicting the successive states in each of the sequences.
\cr
\cr

This method is invoked by several advanced plot functions such as  \code{seqiplot} and \code{seqIplot} (\code{\link{seqplot}} with \code{type="i"} or \code{"I"}), which, in addition to index plots, automatically display the state color legend and allow plotting by group.

When a \code{sortv} variable is provided, the sequences are sorted bottom-up according to its values. With \code{sortv = "from.start"}, sequence are sorted by the elements of the alphabet at the successive positions starting from the beginning of the sequences. Method \code{"from.end"} proceeds similarly, but backward from the last position.

%The interest of sequence index plots has for instance been stressed by \cite{Scherer (2001)} and \cite{Brzinsky-Fay et al. (2006)}.
Index plots of thousands of sequences result in very heavy graphic files when stored in vectorial (PDF or POSTSCRIPT) format because each sequence, even if hidden, is drawn individually. To reduce the file size, we suggest saving the figures in bitmap format by using for instance \code{\link{png}} instead of \code{\link{postscript}} or \code{\link{pdf}}.
}

\seealso{\code{\link{seqplot}}
}

\examples{
## Defining a sequence object with the data in columns 10 to 25
## (family status from age 15 to 30) in the biofam data set
data(biofam)
biofam <- biofam[500:600,] ## using a subsample only
biofam.lab <- c("Parent", "Left", "Married", "Left+Marr",
    "Child", "Left+Child", "Left+Marr+Child", "Divorced")
biofam.seq <- seqdef(biofam, 10:25, labels=biofam.lab)

## Plot of the 10 most frequent sequences
## with bar width proportional to the frequency
plot(biofam.seq)

## Plotting the whole data set
## with no borders
plot(biofam.seq, idxs=0, space=0, border=NA)

## =======
## Weights
## =======
data(ex1)
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)
plot(ex1.seq)
plot(ex1.seq, weighted=FALSE)
}

\author{Gilbert Ritschard}

\keyword{Method}
\keyword{State sequences}
