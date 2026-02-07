\name{seqtreedisplay}
\alias{seqtreedisplay}
\alias{disstreedisplay}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Graphical rendering of a sequence regression tree}
\description{
Generate a graphical representation of a regression tree of state sequence data.
}
\usage{
seqtreedisplay(tree, filename = NULL, seqdata = tree$info$object,
  only.leaf = FALSE, sortv = NULL, diss = NULL, cex.main = 3,
  with.legend = "auto", cex.legend = cex.main, xaxis = FALSE,
  image.format = "png", with.quality = TRUE, cex.quality = cex.main,
  legend.text = NULL, show.tree = TRUE, show.depth = FALSE,
  imgLeafOnly, dist.matrix, title.cex, withlegend, legend.fontsize,
  imageformat, withquality, quality.fontsize, legendtext, showtree,
  showdepth, axes, ...)

disstreedisplay(tree, filename = NULL, image.data= NULL, image.fun = plot,
  only.leaf = FALSE, cex.main = 3, image.format = "png",
  with.quality = TRUE, cex.quality = cex.main,
  legend.text = NULL, show.tree = TRUE, show.depth = FALSE,
  imagedata, imagefunc, imgLeafOnly, title.cex, imageformat,
  withquality, quality.fontsize, legendtext, showtree, showdepth, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tree}{A \code{seqtree} object (as produced by \code{\link{seqtree}}) for \code{seqtreedisplay}. A \code{disstree} object (as produced by \code{\link{disstree}}) for \code{disstreedisplay}.}
  \item{filename}{The name of a file where to save the plot (overwritting existing file). If \code{NULL}, a temporary file is created.}
  \item{seqdata}{The sequence object containing the state sequences plotted in the nodes.}
  \item{only.leaf}{Logical. If \code{TRUE} sequences are plotted only in terminal nodes.}
  \item{sortv}{Argument passed to \code{\link{seqplot}}}
  \item{diss}{Argument passed to \code{\link{seqplot}}}
  \item{cex.main}{Node title magnification. See \code{\link{par}}.}
  \item{with.legend}{Logical. Should the color legend be displayed on the plot?}
  \item{cex.legend}{
    Legend magnification. See \code{\link{par}}. If not specified, use the
    value of \code{cex.main}.
  }
  \item{xaxis}{Logical. Should the x-axis be displayed on the plots? (argument passed to \code{\link{seqplot}})}
  \item{image.format}{Image format of the output file (\code{filename})}
  \item{with.quality}{If \code{TRUE}, a node displaying fitting measures of the tree is added to the plot.}
  \item{cex.quality}{
    Fitting measure text magnification. See \code{\link{par}}. If not specified,
    use the value of \code{cex.main}.
  }
  \item{legend.text}{Character. Optional text information that should be added.}
  \item{show.tree}{Logical. Should the tree be shown on the screen?}
  \item{show.depth}{Logical. If \code{TRUE}, the splits are ordered according to their global pseudo-R2.}
  \item{image.fun}{A function to plot the individuals in a node, see details.}
  \item{image.data}{a \code{data.frame} that will be passed to \code{image.fun}.}
  \item{imgLeafOnly}{Deprecated. Use \code{only.leaf} instead.}
  \item{dist.matrix}{Deprecated. Use \code{diss} instead.}
  \item{title.cex}{Deprecated. Use \code{cex.main} instead.}
  \item{withlegend}{Deprecated. Use \code{with.legend} instead.}
  \item{legend.fontsize}{Deprecated. Use \code{cex.legend} instead.}
  \item{imageformat}{Deprecated. Use \code{image.format} instead.}
  \item{withquality}{Deprecated. Use \code{with.quality} instead.}
  \item{quality.fontsize}{Deprecated. Use \code{cex.quality} instead.}
  \item{legendtext}{Deprecated. Use \code{legend.text} instead.}
  \item{showtree}{Deprecated. Use \code{show.tree} instead.}
  \item{showdepth}{Deprecated. Use \code{show.depth} instead.}
  \item{imagedata}{Deprecated. Use \code{image.data} instead.}
  \item{imagefunc}{Deprecated. Use \code{image.fun} instead.}
  \item{axes}{Deprecated. Use \code{xaxis} instead.}
  \item{\dots}{additional arguments passed to seqplot}
}
\details{
This function generates a tree image. For each node, it invokes \code{\link{seqplot}} for the selected lines of \code{seqdata} as argument.
You should at least specify the \code{type} of the plot to use (\code{type="d"} for instance, see \code{\link{seqplot}} for more details).

The plot is actually not generated as an R plot, but with GraphViz (www.graphviz.org). Hence, \code{seqtreedisplay} only works when GraphViz is correctly installed. If the path to GraphViz is not found, pass the path as a \code{gvpath} argument among the \dots list.

Conversion to image formats other than "jpeg" or "png" is done using ImageMagick (www.imagemagick.org). To use this feature,  ImageMagick (www.imagemagick.org) should hence also be installed.
}
\value{None}

\seealso{
See \code{\link{seqtree}} and \code{\link{disstree}} for examples, and \code{\link{disstree2dot}} for generating "dot" files.
}

\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
\keyword{Plot}
