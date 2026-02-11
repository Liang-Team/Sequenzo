\name{seqedplot}
\alias{seqedplot}
\title{
Graphical representation of a set of events sequences.
}
\description{
This function provides two ways to represent a set of events.
The first one (\code{type="survival"}) plots the survival curves of the first occurrence of each event.
The second one (\code{type="hazard"}) plots the mean counts of each event in the successive periods.
}
\usage{
seqedplot(seqe,
    group=NULL, breaks=20, ages=NULL,
    main="auto", type="survival", ignore=NULL,
	with.legend="auto",cex.legend=1,
    use.layout=(!is.null(group) | with.legend!=FALSE), legend.prop=NA,
    rows=NA, cols=NA,
    xaxis="all", xlab="time",
    yaxis="all",
    ylab=ifelse(type=="survival", "survival probability", "mean number of events"),
    cpal=NULL,
    title, withlegend, axes, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seqe}{an event sequence object as defined by the \code{\link[TraMineR]{seqecreate}} function.}
  \item{group}{Plots one plot for each level of the factor given as argument.}
  \item{breaks}{Number of breaks defining a period.}
  \item{ages}{Two numeric values representing minimum and maximum ages to be represented.}
  \item{main}{String. Title of the graphic. Default is \code{"auto"}, i.e., group levels. Set as \code{NULL} to suppress titles.}
  \item{type}{String. Type of One of \code{"survival"} or \code{"hazard"}. If \code{type="survival"}, survival curves of the first occurrence of each event are plotted. If \code{type="hazard"}, mean numbers of each event in the successive periods are plotted.}
  \item{ignore}{Character vector. An optional list of events that should not be plotted.}
  \item{with.legend}{Logical or string. Defines if and where the legend of the state colors is plotted.
    The default value \code{"auto"} sets the position of the legend automatically.
    Other possible values are \code{"right"} or \code{FALSE}. Obsolete value \code{TRUE} is equivalent to "auto".}
  \item{cex.legend}{expansion factor for setting the size of the font for the labels in the legend. The default value is 1. Values lesser than 1 will reduce the size of the font, values greater than 1 will increase the size.}
  \item{use.layout}{if \code{TRUE}, layout is used to arrange plots when using the group option or plotting a legend.
  When layout is activated, the standard \code{par(mfrow=....)} for arranging plots does not work. With \code{with.legend=FALSE} and \code{group=NULL},
  layout is automatically deactivated and \code{par(mfrow=....)} can be used.}
  \item{legend.prop}{proportion of the graphic area used for plotting the legend when \code{use.layout=TRUE} and \code{with.legend=TRUE}.
  Default value is set according to the place (bottom or right of the graphic area) where the legend is plotted. Values from 0 to 1.}
  \item{rows}{optional arguments to arrange plots when use.layout=TRUE.}
  \item{cols}{optional arguments to arrange plots when use.layout=TRUE.}
  \item{xaxis}{Logical or one of \code{"all"} and \code{"bottom"}. If set as \code{TRUE} or "all" (default value) x-axes are drawn on each plot in the graphic. If set as "bottom" and group is used, x-axes are drawn under the plots of the bottom panel only. If FALSE, no x-axis is drawn.}
  \item{yaxis}{Logical or one of \code{"all"} or \code{"left"}. If set as \code{TRUE} or \code{"all"} (default value) y-axes are drawn on each plot in the graphic. If \code{"left"} and \code{group} is used, the y-axis is displayed on plots of the left panel only. If \code{FALSE} no y-axis is drawn.}
  \item{xlab}{an optional label for the x-axis. If set to \code{NA}, no label is drawn.}
  \item{ylab}{an optional label for the y-axis. If set to \code{NA}, no label is drawn. Can be a vector of labels by group level.}
  \item{cpal}{Color palette used for the events. If \code{NULL}, a new color palette is generated.}
  \item{title}{Deprecated. Use \code{main} instead.}
  \item{withlegend}{Deprecated. Use \code{with.legend} instead.}
  \item{axes}{Deprecated. Use \code{xaxis} instead.}
  \item{\dots}{Additional arguments passed to \code{\link[survival]{plot.survfit}}, \code{\link[survival]{lines.survfit}}, and/or \code{\link[graphics]{legend}}.}
}

\author{Matthias Studer}

\references{
Studer, M., Müller, N.S., Ritschard, G. & Gabadinho, A. (2010), "Classer, discriminer et visualiser des séquences d'événements",
In Extraction et gestion des connaissances (EGC 2010), \emph{Revue des nouvelles technologies de l'information RNTI}. Vol. E-19, pp. 37-48.
}
\examples{
data(actcal.tse)
actcal.tse <- actcal.tse[1:200,]
iseq <- unique(actcal.tse$id)
nseq <- length(iseq)
data(actcal)
actcal <- actcal[rownames(actcal) \%in\% iseq,]
actcal.seqe <- seqecreate(actcal.tse)
seqelength(actcal.seqe) <- rep(12, nseq)
seqedplot(actcal.seqe, type="hazard", breaks=6, group=actcal$sex, lwd=3)
seqedplot(actcal.seqe, type="survival", group=actcal$sex, lwd=3)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{event sequences}
