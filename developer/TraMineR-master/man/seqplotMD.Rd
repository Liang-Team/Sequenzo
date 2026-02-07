\name{seqplotMD}
\alias{seqplotMD}

\title{Plot multidomain state sequences}
\description{
Plot function to render multidomain state sequences by domain and group. The function takes care to make the plots comparable across domains.
}
\usage{
seqplotMD(channels, 
  group = NULL, 
  type = "i", 
  main = NULL, 
  cpal.dom = NULL, 
  missing.color = NULL, 
  ylab = NULL, 
  yaxis = "all", 
  xaxis = "all", 
  xtlab = NULL, 
  stats = "all", 
  cex.axis = 1, 
  with.legend = "auto", 
  ltext.dom = NULL, 
  cex.legend = 1, 
  legend.prop = ifelse(dom.byrow,.25,.15),
  dom.byrow = FALSE, 
  dom.crit = 0, 
  dnames=names(channels), 
  ...)
}
\arguments{
    \item{channels}{List of paired domain state sequence objects.}

    \item{group}{Grouping variable of length equal to the number of multidomain sequences. When not \code{NULL}, a distinct plot is generated for each level of \code{group}.}

    \item{type}{the type of the plot. Available types are 
    \code{"d"} for state distribution plots (chronograms), 
    \code{"dH"} for chronograms with overlayed entropy line, 
    \code{"f"} for sequence frequency plots, 
    \code{"Ht"} for transversal entropy plots, 
    \code{"i"} for index plots of selected sequences, 
    \code{"I"} for index plots of the whole set of sequences, 
    \code{"ms"} for plotting the sequence of modal states,
    \code{"mt"} for mean times plots, 
    \code{"pc"} for parallel  coordinate plots, 
    \code{"r"} for representative sequence plots, and
    \code{"rf"} for relative frequency plots.}

    \item{main}{Character string. Title of the graphic. Default is \code{NULL}.}

    \item{cpal.dom}{List. List of color palettes for the states of each domain. By default, the \code{cpal} attribute of each element of the list \code{channels} is used (see \code{\link{seqdef}}). If user specified, each element of the list should be a vector of colors of length of the alphabet of the corresponding element of \code{channels}.}

    \item{missing.color}{Color for representing missing values inside the sequences. By default, this color is taken for each domain from the \code{missing.color} attribute of the corresponding element of \code{channels}.}

    \item{ylab}{Character string or vector of strings. Optional label of the y-axis. If a vector, y-axis label of each group level. If set as \code{NA}, no label is drawn.}

    \item{yaxis}{Logical or one of \code{"all"} or \code{"left"}. If set as \code{TRUE} or \code{"all"} (default value), sequence indexes are displayed for \code{"i"} and \code{"I"}, mean time values for \code{"mt"}, percentages for \code{"d"} and \code{"f"}, and state/event labels for \code{"pc"}.  If \code{"left"} and \code{group} is used, the y-axis is displayed on plots of the left panel only. If \code{FALSE} no y-axis is drawn. Ignored when \code{type="r"}. For \code{type="f"}, can also be one of \code{"pct"} and \code{"left.pct"}
        }

    \item{xaxis}{Logical or one of \code{"all"} or \code{"bottom"}. If set as \code{TRUE} or \code{"all"} (default value) x-axes are drawn for each plot in the graphic. If set as \code{"bottom"} is used, axes are drawn under the plots of the bottom panel only. If \code{FALSE}, no x-axis is drawn.}

    \item{xtlab}{Vector of length equal to the maximal length of the sequences. Optional labels of the x-axis tick labels. If unspecified, column names of the elements of \code{channels} are used (see \code{\link{seqdef}}).}
        
    \item{stats}{Logical or one of \code{"all"} or \code{"first"}. When \code{type="r"}, if \code{"all"} (default), stats of the MD representatives are displayed on all plots and if \code{"first"}, MD stats are displayed only on the plot of the first domain. Ignored for all non-\code{"r"} types.}   

    \item{cex.axis}{Real value.
      Axis annotation magnification. When \code{type = "r"} and for \code{seqrplot()},
      it also determines the magnification of the plotted text and symbols.
      See \code{\link{par}}.
    }

    \item{with.legend}{Character string or logical. Should legends of the state colors be plotted? Default value \code{"auto"} sets the position of the legend automatically. Value \code{TRUE} is equivalent to \code{"auto"}.}

    \item{ltext.dom}{List. List of vector of character strings of length and order corresponding to the alphabet of the corresponding elements of  \code{channels}. Optional description of the states to appear in the legend. If unspecified, the \code{label} attribute of the corresponding sequence object in \code{channels} is used (see \code{\link{seqdef}}).}
        
    \item{cex.legend}{Real. Legend magnification. See \code{\link{legend}}.}

    \item{legend.prop}{Real in range [0,1]. Proportion of the graphic area devoted to the legends when \code{with.legend=TRUE}. Default value used depends of \code{dom.byrow}.}
        
    \item{dom.byrow}{Logical. Should domains be displayed by row?}

    \item{dom.crit}{Integer in range \code{[-2,ndom]} where \code{ndom} is number of domains. Applies when  \code{sortv} is provided among the \code{...} arguments and is one of \code{"from.start"} or \code{"from.end"}. Domain on which sorting is done. When 0 (default), sorting is done on  multidomain sequences, when -1, sorting is done on first domain then, for ties, successively on the next domains, when -2, sorting is done across domains at each successive position. Does not apply when \code{sortv="mds"} for \code{"rf"} plots.}
        
    \item{dnames}{String vector. Names of the domains.} 

  \item{\dots}{arguments to be passed to the function called to produce the appropriate statistics and the associated plot method (see details), or other graphical parameters. For example, the \code{weighted} argument can be passed to control whether (un)weighted statistics are produced, and \code{with.missing=TRUE} to take missing values into account when computing cross-sectional or longitudinal state distributions. Can also include arguments of \code{\link[graphics]{legend}} such as \code{bty="n"} to suppress the box surrounding the legend.}
}

\details{
The \code{seqplotMD} function is intended for multidomain (or multichannel) sequences, i.e. situations where each case is represented by a sequence in each of several domains. The elements (set of state sequences) of the \code{channels} list are assumed to be paired. Sequences defined in terms of the expanded alphabet formed by the combination of the domain states are called MD sequences. Although the MD sequences (which can be obtained with \code{\link{seqMD}}) may be used for some computations, \code{seqplotMD} does not plot the MD sequences themselves, but the corresponding domain sequences. 

For details on the types of plot, see \code{\link{seqplot}}.

The function organizes the plot area with either domains in rows, groups in columns, and domain legend on the right of each row, or conversely when \code{dom.byrow=FALSE}. Panel titles are defined as "group level: domain", and are preceded by \code{main} when a main value is provided.

The function takes the matching constraints into account as follows:

- For index plots (\code{"i", "I"}), sequences are displayed in same order for each domain.  When a \code{sortv} vector is provided among the \code{...} list, this same sort order is applied to all domains. If a \code{sortv} method is provided, the sorting is computed on the domain specified by the \code{dom.crit} argument. When there is a \code{which.plot} argument (for \code{type="rf"}) in the \code{...} list, it cannot be \code{"both"}.

- Likewise, for relative frequency plots (\code{"rf"}) the same order is retained for all domains. The \code{sortv} argument is handled as for \code{"i"} and \code{"I"}, except for \code{sortv="mds"}, in which case the sorting is done using the provided \code{diss} matrix. 

- For frequency plots (\code{"f"}), sequences displayed correspond to the most frequent MD sequences.

- For plots of representative sequences (\code{"r"}), the representatives are computed for the provided \code{diss} matrix (generally the distances between the MD sequences) and their representation in each domain are displayed. Likewise, for \code{type="rf"}, the medoids of the equally sized groups are computed using the required \code{diss} matrix.

- When \code{type="mt"}, mean time plot, and there is a \code{bar.labels} argument among the \code{...} list, \code{bar.labels} should be a list of vector or matrices of bar labels by domain.

No special handling other than the domain-group titles and the organization of the plot panels is applied to other types (\code{"d","dH","Ht","ms","pc"}).

}

\seealso{\code{\link{seqplot}},
 \code{\link{plot.stslist}},  \code{\link{seqstatd}}, \code{\link{plot.stslist.statd}}, \code{\link{seqtab}}, \code{\link{plot.stslist.freq}}, \code{\link{seqmodst}}, \code{\link{plot.stslist.modst}}, \code{\link{seqmeant}}, \code{\link{plot.stslist.meant}}, \code{\link{seqrep}}, \code{\link{plot.stslist.rep}}, \code{\link{seqrf}}, 
 \code{\link{plot.seqrf}}, \code{\link{seqpcplot}}, \code{\link{seqMD}}.
}


\references{
Ritschard, G., T.F. Liao, and E. Struffolino (2023). Strategies for
multidomain sequence analysis in social research.
\emph{Sociological Methodology}, 53(2), 288-322. \doi{10.1177/00811750231163833}.
}

\examples{
data(biofam)

## Building one channel per type of event (left, children or married)
cases <- 200
bf <- as.matrix(biofam[1:cases, 10:25])
children <-  bf==4 | bf==5 | bf==6
married <- bf == 2 | bf== 3 | bf==6
left <- bf==1 | bf==3 | bf==5 | bf==6

## Building sequence objects
require(colorspace)
m.col <- sequential_hcl(2, palette = "PurpOr")
c.col <- sequential_hcl(2, palette = "Mint")
l.col <- sequential_hcl(2, palette = "OrYel")

child.seq <- seqdef(children, weights=biofam[1:cases,"wp00tbgs"], cpal=c.col)
marr.seq <- seqdef(married, weights=biofam[1:cases,"wp00tbgs"], cpal=m.col)
left.seq <- seqdef(left, weights=biofam[1:cases,"wp00tbgs"], cpal=l.col)
seqdom <- list(LeftHome=left.seq,Marr=marr.seq,Child=child.seq)


seqplotMD(seqdom, type="d", group=biofam[1:cases,"sex"], xaxis="bottom")

seqplotMD(seqdom, type="dH", group=biofam[1:cases,"sex"], dom.byrow=TRUE, 
    xaxis="bottom", yaxis="left")

seqplotMD(seqdom, type="mt", group=biofam[1:cases,"sex"])

seqplotMD(seqdom, type="I", group=biofam[1:cases,"sex"],  
    xaxis="bottom", sortv="from.end")

## sorting on first domain
seqplotMD(seqdom, type="I", group=biofam[1:cases,"sex"],  
    xaxis="bottom", sortv="from.start", dom.crit=1)

seqplotMD(seqdom, type="f", group=biofam[1:cases,"sex"],   
    xaxis="bottom", yaxis="left")

## distances betweem MD sequences
MDseq <- seqMD(seqdom, what="MDseq", ch.sep="+")
diss <- seqdist(MDseq, method="OM", sm="INDELSLOG")

seqplotMD(seqdom, type="rf", group=biofam[1:cases,"sex"],  
    xaxis="bottom", sortv="from.end", dom.crit=0, diss=diss, k=10)

seqplotMD(seqdom, type="r", group=biofam[1:cases,"sex"],  
    xaxis="bottom", dom.crit=0, diss=diss)


}

\author{Gilbert Ritschard}

\keyword{Plot}
