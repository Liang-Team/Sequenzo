\name{seqdef}
\alias{seqdef}
\alias{rbind.stslist}
\alias{summary.stslist}

\title{Create a state sequence object}

\description{
Create a state sequence object with attributes such as alphabet, color palette and state labels. Most TraMineR functions for state sequences require such a state sequence object as input argument. There are specific methods for plotting,  summarizing and printing state sequence objects.
 }

\usage{
seqdef(data, var=NULL, informat="STS", stsep=NULL,
       alphabet=NULL, states=NULL, id=NULL, weights=NULL, start=1,
       left=NA, right="DEL", gaps=NA, missing=NA, void="\%", nr="*",
       cnames=NULL, xtstep=1, tick.last=FALSE, cpal=NULL,
       missing.color="darkgrey", labels=NULL, ...)
}

\arguments{
  \item{data}{a data frame, matrix, or character string vector containing sequence data (tibble will be converted with \code{as.data.frame}).}

  \item{var}{the list of columns containing the sequences. Default is
\code{NULL}, i.e. all the columns. The function detects automatically
whether the sequences are in the compressed (successive states in a
character string) or extended format.}

   \item{informat}{format of the original data. Default is \code{"STS"}.
Other available formats are: \code{"SPS"} and \code{"SPELL"}, in which case the \code{\link{seqformat}} function is called to convert the data into the \code{"STS"} format (see TraMineR user's manual (\cite{Gabadinho et al., 2010}) for a
description of these formats). A better solution is nonetheless to convert first your
data with \code{\link{seqformat}}, so as to have better control over the conversion process and visualize the intermediate \code{"STS"} formatted data.
}

   \item{stsep}{the character used as separator in the original
data if input format is successive states in a character string. If
\code{NULL} (default value), the \code{\link{seqfcheck}} function is
called for detecting automatically a separator among "-" and ":".
Other separators must be specified explicitly.}

  \item{alphabet}{optional vector containing the alphabet (the list of all possible states). Use this option if some states in the alphabet don't appear in the data or if you want to reorder the states. The specified vector MUST contain AT LEAST all the states appearing in the data. It may possibly contain additional states not appearing in the data. If \code{NULL}, the alphabet is set to the distinct states appearing in the data as returned by the \code{\link{seqstatl}} function. See details.}

  \item{states}{an optional vector containing the short state labels. Must have a length equal to the size of the alphabet and the labels must be ordered conformably with alpha-numeric ordered values returned by the \code{\link{seqstatl}} function, or, when \code{alphabet=} is set, with the thus newly defined alphabet.}

  \item{id}{optional argument for setting the rownames of the sequence object. If \code{NULL} (default), the rownames are taken from the input data. If set to \code{"auto"}, sequences are numbered from 1 to the number of sequences. A vector of rownames of length equal to the number of sequences may be specified as well.}

  \item{weights}{optional numerical vector containing weights, which are taken into account by plotting and statistical functions when applicable. Weights must be non-negative and cannot be \code{NA}.}

    \item{start}{starting time. For instance, if sequences begin
at age 15, you can specify 15. At this stage, used only for labelling column names.}

    \item{left}{the behavior for missing values appearing before the
first (leftmost) valid state in each sequence. When \code{NA} (default),
left missing values are treated as 'real' missing values and converted to the
internal missing value code defined by the \code{nr} option. Other
options are \code{"DEL"} to delete the positions containing missing values
or a state code (belonging to the alphabet or not) to replace the
missing values. See \cite{Gabadinho et
al. (2010)} for more details on the options for handling missing
values when defining sequence objects.}

	\item{right}{the behavior for missing values appearing after the
last (rightmost) valid state in each sequence. Same options as for
the \code{left} argument. Default is \code{'DEL'}.}

	\item{gaps}{the behavior for missing values appearing inside the
sequences, i.e. after the first (leftmost) valid state and before the
last (rightmost) valid state of each sequence. Same options as for
the \code{left} argument. Default is \code{NA}.}

	\item{missing}{the code used for missing values in the input
data. Default is \code{NA}. When any other value, all cells containing this value are treated
as \code{NA}s and replaced by \code{nr} or \code{void} code according to the
\code{left}, \code{gaps}, and \code{right} options.}

    \item{void}{the internal
code used by TraMineR for representing void elements in the
sequences. Default is \code{"\%"}. Must be different from \code{left}, \code{gaps}, and \code{right}.}

	\item{nr}{the internal code used by TraMineR for representing real
missing elements in the sequences. Default is \code{"*"}.}

	\item{cnames}{optional names for the columns composing the
sequence data. Those names will be used by default in the graphics as
axis labels. If \code{NULL} (default), names are taken from the
original column names in the data.}

	\item{xtstep}{step between displayed tick-marks and labels on the time x-axis of state sequence plots.
If not overridden by the user, plotting functions retrieve this parameter from the \code{xtstep} attribute of the sequence object. For example, with \code{xtstep=3} a tick-mark is displayed at positions 1, 4, 7, etc...  Default value is 1; i.e., a tick mark is displayed at each position. The display of the corresponding labels depends on the available space and is dealt with automatically.}

  \item{tick.last}{Logical. Should a tick mark be enforced at the last position on the time x-axis?}

	\item{cpal}{an optional color palette for representing the states
in the graphics. If \code{NULL} (default), a color palette is created
by means of the \code{brewer.pal} function of the \code{RColorBrewer}
package for number of states up to 12. When the number of states is less or equal than 8, the
\code{"Accent"} palette is used. If number of states is between 8 and
12, the \code{"Set3"} palette is used. When the number of states is greater than 12,
colors are set using \code{hcl.colors} with
the "Set 3" palette. To specify your own palette use e.g. the \code{\link{colors}}
function, or the RColorBrewer or colorspace packages.}

	\item{missing.color}{alternative color for representing missing
values inside the sequences. Defaults to \code{"darkgrey"}.}

	\item{labels}{optional state labels used for the color legend of
TraMineR's graphics. If \code{NULL} (default), the state names in the
alphabet are used as state labels as well.}

	\item{...}{options passed to the \code{\link{seqformat}} function
for handling input data that is not in STS format.}

}

\details{Applying subscripts to sequence objects (eg. \code{seq[,1:5]
or seq[1:10,]}) returns a state sequence object with some attributes
preserved (alphabet, missing) and some others (start, column names, weights)
adapted to the selected column or row subset. When the number of
columns selected is 1, the returned object is a factor. \cr

For reordering the states use the \code{alphabet} argument. This may
for instance be useful to compare data from
different sources with different codings of similar states. Using
\code{alphabet} permits to order the states conformably in all
sequence objects. Otherwise, the default state order is the
alpha-numeric order returned by the \code{\link{seqstatl}} function
which may differ when you have different original codings.

}

\seealso{
 \code{\link{plot.stslist}} plot method for state sequence objects,\cr
 \code{\link{print.stslist}} print method for state sequence objects,\cr
 \code{\link{is.stslist}} to test whether an object is a proper \code{stslist} object,\cr
 \code{\link{seqplot}} for high level plots of state sequence objects,\cr
 \code{\link{seqecreate}} to create an event sequence object,\cr
 \code{\link{seqformat}} for converting between various longitudinal data formats. }

\value{An object of class \code{stslist}.

There are \code{print}, \code{plot}, \code{rbind}, \code{summary}, and subsetting \code{[,]} methods for such objects.
}

\references{
Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Gabadinho, A., G. Ritschard, M. Studer and N. S. Müller (2010). Mining Sequence Data in \code{R} with the \code{TraMineR}
package: A user's guide. \emph{Department of Econometrics and
Laboratory of Demography, University of Geneva}. }

\examples{
## Creating a sequence object with the columns 13 to 24
## in the 'actcal' example data set
data(actcal)
actcal.seq <- seqdef(actcal,13:24,
	labels=c("> 37 hours", "19-36 hours", "1-18 hours", "no work"))

## Displaying the first 10 rows of the sequence object
actcal.seq[1:10,]

## Displaying the first 10 rows of the sequence object
## in SPS format
print(actcal.seq[1:10,], format="SPS")

## Plotting the first 10 sequences
plot(actcal.seq)

## Re-ordering the alphabet
actcal.seq <- seqdef(actcal,13:24,alphabet=c("B","A","D","C"))
alphabet(actcal.seq)

## Adding a state not appearing in the data to the
## alphabet
actcal.seq <- seqdef(actcal,13:24,alphabet=c("A","B","C","D","E"))
alphabet(actcal.seq)

## Adding a state not appearing in the data to the
## alphabet and changing the states labels
actcal.seq <- seqdef(actcal,13:24,
  alphabet=c("A","B","C","D","E"),
  states=c("FT","PT","LT","NO","TR"))
alphabet(actcal.seq)

## rbind and summary
seq1 <- actcal.seq[1:10,]
seq2 <- actcal.seq[20:25,]
seq <- rbind(seq1,seq2)
summary(seq)

## ============================
## Example with missing values
## ============================
data(ex1)

## With right="DEL" default value
seqdef(ex1,1:13)

## Eliminating 'left' missing values
seqdef(ex1,1:13, left="DEL")

## Eliminating 'left' missing values and gaps
seqdef(ex1,1:13, left="DEL", gaps="DEL")

## ====================
## Example with weights
## ====================
ex1.seq <- seqdef(ex1, 1:13, weights=ex1$weights)

## weighted sequence frequencies
seqtab(ex1.seq)


}

\author{Alexis Gabadinho and Gilbert Ritschard}

\keyword{Data handling}
\keyword{State sequences}
