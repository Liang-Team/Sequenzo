\name{plot.seqrf}
\alias{plot.seqrf}

%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot method for seqrf objects of relative frequency groups of sequences.
}
\description{
Plots the medoids of the RF groups of sequences and/or the boxplots of the distribution of the distances from the sequences to their group medoid.
}
\usage{
\method{plot}{seqrf}(x,
        space=0,
        border=NA,
        which.plot="medoids",
        ylab=NA,
        main=NULL,
        frame.plot=FALSE,
        info="all",
        skipar=FALSE,
        ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{a \code{seqrf} object}
  \item{space}{space between horizontal bars representing medoid sequences (see \code{\link{barplot}})}
  \item{border}{the color to be used for the border of the bars. Use border = NA to omit borders. (see \code{\link{barplot}})}
  \item{which.plot}{string. One of \code{"both"}, \code{"medoids"}, \code{"diss.to.med"}. When \code{"medoids"}, only the index plot of the medoids is displayed, when \code{"diss.to.med"}, the grouped boxplots of the distances to the medoids is displayed, and when \code{"both"} a combined plot of the two plots is displayed.}
  \item{ylab}{string. An optional label for the y-axis. If set as \code{NA} (default), no label is drawn.}
  \item{main}{main graphic title. Default is \code{NULL}.}
  \item{frame.plot}{logical. Should a frame be plotted around the grouped boxplots?}
  \item{info}{string. One of \code{"all"} (default), \code{"subtitle"}, \code{"stat"}, and \code{"none"}.}
  \item{skipar}{logical. Should \code{plot} skip internal \code{par} and \code{layout} commands. May be necessary to combine \code{"medoids"} or \code{"diss.to.med"} plots with \code{layout} or \code{par(mfrow=...)}}
  \item{...}{further arguments passed to \code{\link{plot.stslist}} including graphical parameters (\code{\link{par}})}
}

\details{
The plot of the medoids is generated with the plot method for \code{stslist} objects and the boxplots with a slightly adapted version of the boxplot function for weighted data (\code{wtd.boxplot}) of the ENmisc package by Erich Neuwirth.

When \code{which.plot="both"}, \code{\link{layout}} is used to produce the combined plot. Therefore, the resulting combined plot cannot be nested into other combinations using either \code{par(mfrow=...)} or \code{layout}. To combine with other plots, you must produce separately the plot of the medoids and the grouped boxplots using successively \code{which.plot="medoids"} and \code{which.plot="diss.to.med"}.

When arguments \code{xaxis} and \code{yaxis} are on the \code{...} list to be passed to \code{\link{plot.stslist}}, they are also used to control the display of the x and y axis of the boxplot.

With \code{info="stat"} or \code{"all"}, the pseudo R2 and F statistics are displayed under the plot, but only when \code{which.plot = "both"}.
}

\seealso{\code{\link{seqrf}}, \code{\link{seqrfplot}}
}

\author{
Gilbert Ritschard.
}



