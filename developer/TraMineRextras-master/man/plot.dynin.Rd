\name{plot.dynin}
\alias{plot.dynin}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Dynamic index plot
%%  ~~function to do ... ~~
}
\description{Plot of dynamic (i.e. successive) cross-sectional summaries of an individual index. The successive values of the individual index for all sequences should be collected in a \code{dynin} table as produced by \code{\link{seqindic.dyn}}.
%%  Generic function for
}
\usage{
\method{plot}{dynin}(x, fstat=weighted.mean, group=NULL, conf=FALSE,
     main="auto", col=NULL, lty=NULL, lwd=3.5, ylim=NULL,
     ylab=NULL, xlab=NULL, xtlab=NULL, xtstep=NULL, tick.last=NULL,
     with.legend=TRUE, glabels=NULL, legend.pos="topright",
     horiz=FALSE, cex.legend=1, bcol=NULL, na.rm=FALSE, ret=FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{object of class \code{dynin} as produced by \code{\link{seqindic.dyn}}
}
  \item{fstat}{function: summary function to compute the values plotted. Default is \code{\link{weighted.mean}} with weights taken from the \code{weights} attribute of \code{x}. When \code{\link{weighted.mean}} and \code{x} has no weights, \code{mean} is used instead.
}
  \item{group}{factor or discrete vector: group membership; a curve is drawn for each group. If \code{NULL} (default) a single curve for the whole set is drawn.
}
  \item{conf}{logical or numeric: If logical, should confidence bands be displayed? If numeric, confidence probability. \code{TRUE} is equivalent to .95. Applies only when \code{fstat=mean} or \code{fstat=weighted.mean}.
}
  \item{main}{character string: Plot title. Default is "auto" that prints a default title. Set as \code{NULL} to suppress the title.
}
  \item{col}{color vector. Group line colors. If \code{NULL} (default), colors are automatically assigned using \code{qualitative_hcl} with the \code{'Dark 3'} palette (see \code{\link[colorspace]{hcl_palettes}}).
}
  \item{lty}{string vector. Group line types (see \code{\link{lines}}). If \code{NULL} (default), types are automatically assigned.
}
  \item{lwd}{integer vector: Group line widths (see \code{\link{lines}}). If \code{NULL} (default), set as 3.5.
}
  \item{ylim}{pair of numerics defining the range for the y-axis. If left \code{NULL}, the limits are defined from the data.}

  \item{ylab}{character string: y axis label.
  }
  \item{xlab}{character string: x axis label.
  }

  \item{xtlab}{vector of strings defining the x-axis tick labels. If \code{NULL}, column names of the \code{x} table are used.}

  \item{xtstep}{integer: step between tick marks on the x-axis. If unspecified, attribute \code{xtstep} of the \code{x} object is used.}

  \item{tick.last}{logical. Should a tick mark be enforced at the last position on the x-axis? If unspecified, attribute \code{tick.last} of the \code{x} object is used.}

  \item{glabels}{a vector of strings with the curve labels. If \code{NULL} curves are labeled with the levels of the group variable}

  \item{with.legend}{logical: Should the legend be plotted. Default is \code{TRUE}.}

  \item{legend.pos}{legend position: default is \code{"topright"}. See\code{\link{legend} }.}

  \item{horiz}{logical: Should the legend be displayed horizontally. Set as \code{FALSE} by default, i.e., legend is displayed vertically.}

  \item{cex.legend}{Scale factor for the legend.}

  \item{bcol}{color vector. For confidence bands. If \code{NULL} (default), colors are automatically assigned using \code{qualitative_hcl}  with the \code{'Pastel 1'} palette (see \code{\link[colorspace]{hcl_palettes}}).
  }

  \item{na.rm}{logical. When \code{fstat} is \code{mean} or \code{weighted.mean}, should \code{NA}'s be stripped before computation? Ignored for any other \code{fstat} function.}

  \item{ret}{logical: Should the plotted values be returned?}

  \item{...}{additional plot parameters (see \code{\link{par}}).}
}
\details{
Together with \code{\link{seqindic.dyn}} this function implements the dynamic sequence analysis approach of \cite{Pelletier et al. (2020)}.
%%  ~~ If necessary, more details than the description above ~~

The function first computes the summary table using the \code{fstat} function. Each row of the summary table is then plotted as a line, except rows that contain \code{NA}s. Setting \code{na.rm=TRUE} helps sometimes to prevent some \code{NA}s in the summary table.

Confidence bands are computed for a confidence level of 95\% and assuming a normal distribution.

}

\value{
  If \code{ret=TRUE}, a matrix with the successive group summaries (One row per group) and, when \code{conf=TRUE}, the matrices with the lower and upper bounds of the confidence intervals as attributes \code{L.grp} and \code{U.grp}.
}

\references{
Pelletier, D., Bignami-Van Assche, S., & Simard-Gendron, A. (2020) Measuring Life Course Complexity with Dynamic Sequence Analysis, \emph{Social Indicators Research} \doi{10.1007/s11205-020-02464-y}.

%% ~put references to the literature/web site here ~
}
\author{Gilbert Ritschard
}

\seealso{
See Also \code{\link{seqindic.dyn}} (with examples)
}
\examples{
## See examples on 'seqindic.dyn' help page
}
