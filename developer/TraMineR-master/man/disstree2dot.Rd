\name{disstree2dot}
\alias{disstree2dot}
\alias{disstree2dotp}
\alias{seqtree2dot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Graphical representation of a dissimilarity tree}
\description{
Functions to generate a "dot" file and associated images files that can be
used in GraphViz to get a graphical representation of the tree. }
\usage{
disstree2dot(tree, filename, digits = 3, image.fun = NULL, image.data = NULL,
  only.leaf = FALSE, device = "jpeg", image.format = "jpg",
  device.args = list(), use.title = TRUE, label.pos = "main",
  node.pos = "main", split.pos = "sub", cex.main = 1,
  legend.text = NULL, image.legend = NULL, image.quality = NULL,
  show.depth = FALSE, title.outer = FALSE,
  imagefunc, imagedata, imgLeafOnly, devicefunc, imageext,
  device.arg, label.loc, node.loc, split.loc, title.cex, legendtext,
  legendimage, qualityimage, showdepth, ...)

disstree2dotp(tree, filename, image.data = NULL, only.leaf = FALSE,
  image.fun = plot, cex.main = 3, with.quality = TRUE,
  cex.quality = cex.main, title.outer = FALSE,
  imagedata, imgLeafOnly, imagefunc, title.cex, withquality,
  quality.fontsize, ...)

seqtree2dot(tree, filename, seqdata = tree$info$object, only.leaf = FALSE,
  sortv = NULL, diss = NULL, cex.main = 3, with.legend = "auto",
  cex.legend = cex.main, with.quality = FALSE,
  cex.quality = cex.main, xaxis = FALSE,
  imgLeafOnly, dist.matrix, title.cex,
  withlegend, withquality, axes,  ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tree}{The tree to be plotted.}
  \item{filename}{A filename, without extension, that will be used to generate image and dot files.}
  \item{digits}{Number of significant digits to plot.}
  \item{image.fun}{A function to plot the individuals in a node, see details.}
  \item{image.data}{a \code{data.frame} that will be passed to \code{image.fun}, see details.}
  \item{only.leaf}{Logical: If \code{TRUE}, only terminal node will be plotted.}
  \item{device}{A device function, \code{"jpeg"} by default.}
  \item{image.format}{extension for image files.}
  \item{device.args}{Argument passed to \code{device}.}
  \item{use.title}{Logical: If \code{TRUE}, node information will be printed using \code{\link{title}} command, see details.}
  \item{label.pos}{Location of the node label, see \code{\link{title}} for possible values.}
  \item{node.pos}{Node content location, see \code{\link{title}} for possible values.}
  \item{split.pos}{Split information location, see \code{\link{title}} for possible values.}
  \item{cex.main}{\code{cex} applied to all calls to \code{title} (see \code{use.title}).}
  \item{title.outer}{Logical: If \code{TRUE}, the title (see \code{use.title}) is printed in the outer margins.}
  \item{legend.text}{An optional text appearing in a distinct node.}
  \item{image.legend}{An optional image file appearing in a distinct node.}
  \item{image.quality}{An optional image file appearing in a distinct node.}
  \item{show.depth}{Logical. If \code{TRUE}, information about depth of the tree is added to the plot.}
  \item{with.quality}{If \code{TRUE}, a node displaying fitting measures of the tree is added to the plot.}
  \item{cex.quality}{Numeric. Size of the font of the fitting measures node.}
  \item{seqdata}{a sequence object as defined by the the \code{\link{seqdef}} function.}
  \item{sortv}{The name of an optional variable used to sort the data before plotting, see \code{\link{seqplot}}.}
  \item{diss}{The name of an optional dissimilarity matrix used to find representative sequences, \code{\link{seqrplot}}.}
  \item{with.legend}{defines if and where the legend of the state colors is plotted. The default value \code{"auto"} sets the position of the legend automatically. Other possible value is \code{"right"}. Obsolete value \code{TRUE} is equivalent to \code{"auto"}.}
  \item{cex.legend}{Size of the font of the legend.}
  \item{xaxis}{Logical. Should the x-axis be drawn on the plots?}
  \item{imagefunc}{Deprecated. Use \code{image.fun} instead.}
  \item{imagedata}{Deprecated. Use \code{image.data} instead.}
  \item{imgLeafOnly}{Deprecated. Use \code{only.leaf} instead.}
  \item{devicefunc}{Deprecated. Use \code{device} instead.}
  \item{imageext}{Deprecated. Use \code{image.format} instead.}
  \item{device.arg}{Deprecated. Use \code{device.args} instead.}
  \item{label.loc}{Deprecated. Use \code{label.pos} instead.}
  \item{node.loc}{Deprecated. Use \code{node.pos} instead.}
  \item{split.loc}{Deprecated. Use \code{split.pos} instead.}
  \item{title.cex}{Deprecated. Use \code{cex.main} instead.}
  \item{legendtext}{Deprecated. Use \code{legend.text} instead.}
  \item{legendimage}{Deprecated. Use \code{image.legend} instead.}
  \item{qualityimage}{Deprecated. Use \code{image.quality} instead.}
  \item{showdepth}{Deprecated. Use \code{show.depth} instead.}
  \item{withquality}{Deprecated. Use \code{with.quality} instead.}
  \item{quality.fontsize}{Deprecated. Use \code{cex.quality} instead.}
  \item{dist.matrix}{Deprecated. Use \code{diss} instead.}
  \item{withlegend}{Deprecated. Use \code{with.legend} instead.}
  \item{axes}{Deprecated. Use \code{xaxis} instead.}
  \item{\dots}{other parameters that will be passed to \code{image.fun} or \code{\link{seqplot}} (for \code{seqtree2dot}).}
}
\details{
  These functions generate a "dot" file that can be used in GraphViz (\code{http://www.graphviz.org}).
It also generates one image per node through a call to
\code{image.fun} passing the selected lines of \code{image.data} if
present or otherwise a list of indexes (of individuals belonging to a
node). These functions are not intended to be used by end-user. See \code{\link{seqtreedisplay}}
and \code{\link{disstreedisplay}} for a much simpler way to generate a graphical representation of a
tree (\code{\link{seqtree}} or \code{\link{disstree}}).

If the path to GraphViz is not found, pass the path as \code{gvpath} argument among the \dots list.

\code{seqtree2dot} is a shortcut for sequences objects using the plot function \code{\link{seqplot}}. For each node, it calls \code{\link{seqplot}} with the corresponding subset of rows of \code{seqdata} and the provided \code{\link{seqplot}}'s arguments.
    You should at least specify the \code{type} of the plot (e.g.  \code{type="d"}, see \code{\link{seqplot}} for more details).


If \code{use.title} is \code{TRUE}, \code{image.fun} should take care
to leave enough space for the title.

\code{disstree2dotp} is a simplified interface of
\code{disstree2dot} which automatically leaves enough space for the title and subtitles. These functions are intended to be generic.


}
\value{
  Nothing but generates a "dot" and several image files (one per node) in the current working directory (see \code{\link{getwd}} and \code{\link{setwd}}).
}

\seealso{\code{\link{seqtree}} and \code{\link{seqtreedisplay}}, \code{\link{disstree}} and \code{\link{disstreedisplay}}.}
\author{Matthias Studer (with Gilbert Ritschard for the help page)}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Dissimilarity-based analysis}
\keyword{Plot}
