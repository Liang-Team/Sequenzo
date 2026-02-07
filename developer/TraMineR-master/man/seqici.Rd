\name{seqici}
\alias{seqici}
\title{Complexity index of individual sequences}
\description{ Computes the complexity index, a composite measure of
sequence complexity. The index uses the number of transitions in the
sequence as a measure of the complexity induced by the state ordering
and the longitudinal entropy as a measure of the complexity induced
by the state distribution in the sequence. }

\usage{
 seqici(seqdata, with.missing=FALSE, silent=TRUE)
 }

\arguments{
  \item{seqdata}{a sequence object as returned by the the \code{\link{seqdef}} function.}

  \item{with.missing}{if set to \code{TRUE}, missing status (gaps in sequences)
is handled as an additional state when computing the state
distribution and the number of transitions in the sequence.}

  \item{silent}{logical: should messages about running operations be displayed?}
 }

\details{
 The \emph{complexity index} \eqn{C(s)} of a sequence
\eqn{s} is
 \deqn{ C(s)= \sqrt{\frac{q(s)}{q_{max}} \,\frac{h(s)}{h_{max}}} }{%
 C(s) = ([q(s) h(s)]/[q_\{max\} h_\{max\}])^(.5)}

where \eqn{q(s)} is the number of transitions in the sequence,
\eqn{q_{max}} the maximum number of transitions,
\eqn{h(s)} the within entropy, and \eqn{h_{max}} the theoretical maximum
entropy which is \eqn{h_{max} = -\log 1/|A|}{h_max = -log 1/|A|} with \eqn{|A|} the size of the alphabet.

The index \eqn{C(s)} is the geometric mean of its two normalized components and is,
therefore, itself normalized.
The minimum value of 0 can only be reached by a
sequence made of one distinct state, thus containing 0 transitions
and having an entropy of 0. The maximum 1 of \eqn{C(s)} is reached
when the two following conditions are fulfilled: i) Each of the state
in the alphabet is present in the sequence, and the total durations
are uniform, i.e. each state occurs \eqn{\ell/|A|}{l/|A|} times, and ii) the number
of transitions in the sequence is \eqn{\ell-1}{l-1}, meaning that the length \eqn{\ell_d}{l_d} of the DSS is equal to the length of the sequence \eqn{\ell}{l}.
 }

\value{
 a single-column matrix of length equal to the number of sequences in
\code{seqdata} containing the complexity index value of each
sequence.
 }

\seealso{
 \code{\link{seqindic}}, \code{\link{seqient}}, \code{\link{seqipos}}.

 For alternative measures of sequence complexity see \code{\link{seqST}},  \code{\link{seqivolatility}}.
}

\references{

Gabadinho, A., G. Ritschard, N. S. Müller and M. Studer (2011). Analyzing and Visualizing State Sequences in R with TraMineR. \emph{Journal of Statistical Software} \bold{40}(4), 1-37.

Gabadinho, A., Ritschard, G., Studer, M. and Müller,
N.S. (2010). "Indice de complexité pour le tri et la comparaison de
séquences catégorielles", In \emph{Extraction et gestion des
connaissances (EGC 2010), Revue des nouvelles technologies de
l'information RNTI}. Vol. E-19, pp. 61-66.

Ritschard, G. (2023), "Measuring the nature of individual sequences", \emph{Sociological Methods and Research}, 52(4), 2016-2049. \doi{10.1177/00491241211036156}.
}
\examples{
## Creating a sequence object from the mvad data set
data(mvad)
mvad.labels <- c("employment", "further education", "higher education",
                    "joblessness", "school", "training")
mvad.scodes <- c("EM","FE","HE","JL","SC","TR")
mvad.seq <- seqdef(mvad, 15:86, states=mvad.scodes, labels=mvad.labels)

##
mvad.ci <- seqici(mvad.seq)
summary(mvad.ci)
hist(mvad.ci)

## Example using with.missing argument
data(ex1)
ex1.seq <- seqdef(ex1, 1:13)
seqici(ex1.seq)
seqici(ex1.seq, with.missing=TRUE)
}

\author{Alexis Gabadinho (with Gilbert Ritschard for the help page)}

\keyword{Longitudinal characteristics}
