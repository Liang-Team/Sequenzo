\name{NEWS}
\title{NEWS file for the TraMineR package}
\section{Changes in version 2.2-13 [2025-12-14] }{

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{plot.stslist()}, plot method for object of class \code{stslist}, 
            gains new attributes \code{sampling} and \code{sample.meth} to plot only a subsample 
            of the sequences. Handy when large number of sequences must be plotted.
            \item
            \code{dissrf()}, weights are normalized to sum up to the number of cases and, when k=1, computation is skipped and the overall medoid of the data is returned.
            \item
            \code{dissrf()}, explicit warnings when negative degrees of freedom.
        }
    }
    
    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{plot.stslist()}, plot method for object of class \code{stslist}: bad weight     assignment when \code{idxs} is not \code{seq(1, to=length(idxs))}.
            \item
            Subsetting state sequence (\code{stslist}) objects with \code{[]} did not work when selecting columns with column names.
        }
    }
    
    \subsection{Misc.}{
        \itemize{
            \item
            Authors added in description of provided data sets.
            \item
            Examples added in description of \code{biofam}, \code{mvad}, \code{actcal}, and \code{famform} data sets.
            \item
            Fixed outdated and non-canonical URLs in vignette.
        }
    }
}


\section{Changes in version 2.2-12 [2025-06-29] }{

    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{dissdomassoc()}: issue with ties when ranking values for spearman correlation.
            \item  
            \code{disscenter()}: wrong distance values returned when \code{allcenter=TRUE}.
            \item
            \code{seqformat()}: useless test on \code{covar} when \code{to} is not \code{"SRS"}.
            \item
            \code{seqformat()} with \code{from="SPELL"} (\code{SPELL_to_STS}): unreadable 
            list of cases with spells ending before start time.
            \item
            \code{seqformat()}: several internal \code{checkindexes()} badly specified.
        }
    }
}


\section{Changes in version 2.2-11 [2024-12-08] }{

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{dissrep()}: when all distances are zero, the
            number \code{nb} of cases in the neighborhood of the
            representative is set as the sum of weights
            (was zero in previous versions)
        }
    }

    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{dissrep()} error when seqdata is a single sequence.
            \item
            \code{seqecreate()} (and \code{tmrsequenceseveral} cpp function): segfault error when \code{intEvent} is \code{NULL}. (Fix required by Brian Ripley)
        }
    }
}

\section{Changes in version 2.2-10 [2024-05-20] }{

    \subsection{New functions}{
        \itemize{
            \item
            \code{dissmergegroups()} merges groups so as to minimize loss of partition quality.
            \item
            \code{seqhasmiss()} count and identification of sequences with missings.
        }
    }

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{seqdist()} new argument \code{opt.args} to test
            non-documented arguments at development stage.
            \item
            \code{seqdef()} now raises an error when it finds negative or \code{NA} weights.
            \item
            \code{seqprecstart()} includes derived state.order and state.equiv as attributes of returned object.
        }
    }

    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{seqcost()} \code{method="INDELSLOG"} returned \code{NA} costs for token 0 when alphabet is numeric.
            \item
            \code{print.stslist(, format="SPS")} error with some version of R.
            \item
            \code{plot.stslist.statd(type="dH")} error when \code{xtstep > 1}.
            \item
            \code{plot.stslist.statd(type="Ht")} argument \code{main} was ignored.
            \item
            \code{plot.stslist.statd(type="Ht")} displayed useless x-label "Index" when \code{xlab=NULL}.
            \item
            \code{seqdHplot()} and \code{seqplot(,type="dH")} use \code{col.entr}
    instead of \code{col} to control color of overlayed entropy
    because of name conflict.
            \item
            \code{seqidegrad()} and \code{seqinsecurity}: unexpected results when order of undesirableness degrees provided with \code{stprec} differed from alphabet order.
            \item
            \code{seqdomassoc()} error when only two domains.
            \item
            noRemap issues.
        }
    }
}

\section{Changes in version 2.2-9 [2024-01-08] }{

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            All TraMineR plot functions: improved support of graphical parameters. (E.g., \code{las} and \code{cex.axis} can now be used to control the display of tick labels.)
            \item
            \code{seqdist()} transforms \code{kweights} into vector \code{rep(kweights,ncol(seqdata))} when it is scalar.
            \item
            \code{seqdist()} raises a warning when the vector \code{kweights} is shorter than the number of columns in \code{seqdata}.
        }
    }
    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{seqdef()}: lack of separation between found tokens not listed in the alphabet.
            \item
            \code{seqdist()}: error when all DSS have length 1 and \code{method} is one of OMspell, NMSMST, and SVRspell.
            \item
            \code{seqdist()}: error when all DSS are identical with length 1 and \code{method} is OMstran.
            \item
            \code{seqdist()} with \code{method="OMstran"}: returned distance matrix was not labeled with sequence rownames.
            \item
            Issues with lost braces in .Rd files (NOTE in CRAN check)
        }
    }
}

\section{Changes in version 2.2-8 [2023-09-18] }{

    \subsection{New functions}{
        \itemize{
            \item
            \code{disstree.get.rules()} character string vector of the classification rules of a \code{disstree} as R commands.
            \item
            \code{disstree.assign()} determines the index of the rules that apply to the provided profiles.
        }
    }

    \subsection{Changes in exiting functions}{
        \itemize{
            \item
            \code{seqST()} When number \code{phi} of subsequences is \code{NaN}, it is set as \code{.Machine$double.xmax} to compute turbulence and a warning is raised.
            \item
            \code{seqplot()} and \code{seqplotMD()}: when \code{type="f"}, the \code{yaxis} argument can now also be \code{"pct"} or \code{"left.pct"}.
            \item
            \code{seqrep()} and \code{dissrep()}: returned object gains attributes \code{Rep.group} with the representative to which each object is assigned, and \code{idx.rep} a list with indexes of all occurrences of each representative.
            \item
            \code{seqtreedisplay()} and \code{seqtree2dot()}: argument \code{axes} is deprecated and replaced by \code{xaxis} to be conformable with \code{seqplot}.
            \item
            \code{disstreedisplay()} and \code{seqtreedisplay()} now raise an error when the file \code{filename} of the image cannot be created.
            \item
            \code{disstreeleaf()} gains an argument \code{collapse} to control separator within classes of categories.
        }
    }

    \subsection{Bug fixes}{
         \itemize{
            \item
            \code{seqfplot()} and \code{seqplot(type="f")}: option \code{yaxis="pct"} did no longer work after update to v 2.2-7.
            \item
            \code{disstreedisplay()} and \code{seqtreedisplay()} did not find the GraphViz path.
            \item
            \code{disstreedisplay()} and \code{seqtreedisplay()} uselessly printed \code{TRUE} when filename was not \code{NULL}.
         }
    }
}

\section{Changes in version 2.2-7 [2023-03-31] }{

    \subsection{New functions}{
        \itemize{
            \item
            \code{seqplotMD()} plot of multidomain/multichannel sequences by domain and group.
            \item
            \code{seqdomassoc()} pairwise state association between domains.
        }
    }

    \subsection{Changes in exiting functions}{
        \itemize{
            \item
            \code{dissrf()} and \code{seqrf()} \code{sortv="mds"} is now the default and \code{sortv=NULL} selects the original seqdata order.
            \item
            \code{seqrfplot()} (\code{seqplot(..., type="rf", ...)}) with option \code{sortv="mds"} (new default in \code{seqrf}) the MDS sorting is done on the whole seqdata instead of independently for each group.
            \item
            \code{seqMD()}: \code{with.missing} can now be \code{NULL} (new default), in which case the with.missing value of each domain is set as \code{TRUE} when the domain has missings and as \code{FALSE} otherwise.
            \item
            \code{seqMD()}: new logical argument \code{fill.with.miss}, which offers the option to fill the shorter sequences with missings in case of non-matching lengths across domains.
            \item
            \code{seqplot()} and all the \code{seqxxplot()} family functions: \code{ylab} can be a vector of labels by group level.
            \item
            \code{seqplot()} and all the \code{seqxxplot()} family functions support arguments of \code{legend} among the \code{...} list.
            \item
            \code{"[.stslist"} raises a warning when row subsetting condition has \code{NA}s.
        }
    }

    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{seqrf()}: options \code{sortv="from.start"} and \code{sortv="from.end"} did not work as expected.
            \item
            \code{seqMD()}: error when sequences had different lengths.
            \item
            foreign c function \code{wcorr} called with character string naming while package calls       \code{R_forceSymbols}. (Fix asked for by K. Hornik).
        }
    }

    \subsection{Misc.}{
        \itemize{
            \item
            Statement "SystemRequirements: C++11" dropped from DESCRIPTION as requested by CRAN check
            \item
            Using new CITATION commands.
        }
    }

}


\section{Changes in version 2.2-6 [2022-11-29] }{

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{seqplot()} and all the \code{seqxxplot()} family functions: default value for the \code{main} argument is now \code{"auto"} and \code{NULL} removes all titles as it should.
            \item
            \code{seqplot()} and all the \code{seqxxplot()} family functions: argument \code{axes} is deprecated and replaced by \code{xaxis}. Argument \code{yaxis} can now also be one of \code{"all"} or \code{"left"}.
            \item
            \code{plot()} methods for objects returned by \code{seqdef}, \code{seqstatd}, \code{seqtab}, \code{seqmodst}, \code{seqmeant}, and \code{seqrf} now display the \code{ylab} closer from the y-axis when \code{yaxis=FALSE}.
            \item
            \code{plot()} method for object returned by \code{seqrep} now displays the \code{ylab} closer from the y-axis when \code{stats=FALSE}.

        }
    }
    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{summary.dissrf()}: five number statistics ignored weights (reported by Marcel Raab)
            \item
            \code{seqstatl()}: non-integer state value were not recognized.
            \item
            \code{plot()} methods for objects returned by \code{seqdef}, \code{seqstatd}, \code{seqtab}, \code{seqmodst}, \code{seqmeant}, \code{seqrf}, and \code{seqrep} ignored the \code{cex.lab} argument. \item
            issues with prototypes in init.c
            \item
            removed unused code with deprecated \code{sprintf} in eventdictionary.cpp
        }
    }

    \subsection{Misc.}{
        \itemize{
            \item
            Replaced \code{seqdistmc} with \code{seqMD} in examples of \code{dissdomassoc}.
        }
    }
}

\section{Changes in version 2.2-5 [2022-08-30] }{

    \subsection{New functions}{
        \itemize{
            %\item
            %\code{seqplotMD()} plot of multidomain sequences.
            \item
            \code{dissrf()} medoids of relative frequency (RF) groups
            \item
            \code{seqrf()} medoid sequences of RF groups
            \item
            \code{plot.seqrf()} plot method for medoid sequences of RF groups
            \item
            \code{seqrfplot()} alias for \code{seqplot(...,type="rf",...)}
            \item
            print and summary methods for \code{dissrf} and \code{seqrf} objects
            \item
            \code{seqfposend()} position of end of spell in given state
        }

    }

    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{seqplot()} new value \code{"rf"} (RF plots) for argument type.
            \item
            \code{seqtab()} result gains an attribute \code{idxf} with indexes of first found frequent sequences.
            \item
            \code{seqdistmc()} renamed \code{seqMD} to better reflect that the function can return various outcomes related to multidomain (MD) sequences, not just distances based on the MCSA trick. Old name maintained as an alias for backward compatibility.
            \item
            \code{plot.stslist()} now raises an explicit error for bad \code{ytlab} value.
        }
    }

    \subsection{Bug fixes}{
        \itemize{
          \item
          \code{plot.stslist()} possible conflict between \code{x} names and arguments of \code{order}.
          \item
          \code{plot.stslist.rep()} bad position of y-axis when \code{!is.na(ylab)}
        }
    }

    \subsection{Misc.}{
        \itemize{
            \item
            Proper \code{wtd.cor}, \code{wtd.mean}, \code{wtd.var}, \code{wtd.fivenum}, and
            \code{wtd.boxplot}  internal functions to avoid
            issues with dependence on packages weights, Hmisc, and ENmisc. (Code in TraMineR-weights.R and TraMineR-wtd-stats.R.)
            \item
            Function \code{TraMineR.checkupdates} has been removed.
        }
    }
}

\section{Changes in version 2.2-4 [2022-06-08] }{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqdHplot()} alias for \code{seqplot(...,type="dH",...)}
      \item
      \code{seqlength.align()} sets lengths of sequences of multiple domains as the shortest length among the domains.
      \item
      \code{seqmaintokens()} returns indexes of more frequent tokens.
    }
  }
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{plot.stslist.statd()} new type \code{"dH"} that overlays entropy line on the chronogram. In addition, new arguments \code{col}, \code{lwd}, and \code{ylim} allow controlling aspect of entropy line.
      \item
      \code{seqplot()} new type \code{"dH"}.
      \item
      \code{seqformat()}, \code{seqdef()}, \code{seqdecomp()}, \code{seqconc()}, \code{seqstatl()} now support tibble input data.
      \item
      \code{seqformat()} accepts a tibble as \code{pdata} table.
      \item
      \code{seqformat()} accepts a character string vector as input data.
      \item
      \code{seqdistmc()}: option \code{what="sm"} becomes \code{what="cost"}.
      Deprecated \code{"sm"} value maintained for backward compatibility.
      \item
      \code{plot.stslist.rep()} gains two arguments \code{seq.alt} to specify
      the domain in which representatives are displayed and \code{info} to control
      whether coverage info should be displayed.
    }
  }
  \subsection{Misc.}{
    \itemize{
      \item
      Suppressed dependence on cNORM because of license
      incompatibility. (\code{dissdomassoc} uses now a proper
      internal function in place of \code{cNORM:::weighted.rank}).
    }
  }
}


\section{Changes in version 2.2-3 [2022-01-19] }{

  \subsection{New functions}{
    \itemize{
      \item
      \code{summary.ddomassoc()} summary method that organizes outcome of \code{dissdomassoc} in table form.
    }
  }

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{dissdomassoc()} now returns also p-values of computed correlations.
      \item
      \code{plot.subseqelistchisq()} (plot of outcome of \code{seqecmpgroup}) gains attribute \code{with.legend}.
      \item
      \code{plot.stslist.modst()} new attribute \code{info} to control whether frequency of sequence of modal state should be displayed.
      \item
      \code{plot.ststlist.meant()} now uses an internal function
      to draw error bars to avoid issues with dependence on Hmisc.
      \item
      \code{plot.ststlist.meant()} new arguments \code{bar.labels},
      \code{cex.barlab}, and \code{offset.barlab} to display and control bar labels.
      \item
      \code{seqmtplot()} (\code{seqplot(..,type="mt", ..)}) adapted for
       handling bar labels with groups. See details in \code{seqplot} help page.
      \item
      \code{seqformat()} with \code{from="SPELL"}, provides info about number of sequences truncated, empty sequences because of spell occurring before birth year, and empty because of missing start time. A new attribute \code{issues} contains
      the indexes of sequences with issues.
      \item
      \code{seqdef()} now prints, when applicable, the number of empty sequences and lists the first 9 empty sequences only.
      \item
      \code{seqdistmc()} gains an argument \code{ch.sep} to control the separator symbol for building state names of the expanded alphabet.
      \item
      \code{seqdistmc()} rows and columns of the returned distance matrix and the mc sequences are now labelled with rownames of the
      first channel. Rows and columns of the returned mc substitution cost matrix are labelled with the expanded alphabet.
      \item
      \code{seqdistmc()} columns of the returned expanded sequence object now labelled with column names of longest domain.
      \item
      \code{seqdistmc()} \code{indel} can now be \code{"auto"} (new default) to automatically set indel values. In addition, state dependent indels are now supported.
      \item
      \code{seqdistmc()} \code{with.missing} can now be a vector to provide different values for each channel.
      \item
      \code{seqdistmc()} now also supports \code{"INDELS"} and \code{"INDELSLOG"} values for the \code{sm} argument.
      \item
      \code{seqcost()} removed "\code{->}" at end of row/column labels of matrix \code{sm}.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqdiff()} negative values in \code{cmprange} caused an error.
      \item
      \code{seqdiff()} changed deprecated \code{norm=TRUE} into \code{norm='auto'} in default \code{seqdist.args} value.
      \item
      \code{seqecreate()} \code{NA}s introduced by coercion with character \code{id}.

    }
  }

  \subsection{Misc.}{
    \itemize{
      \item
      Dependence on boot and RColorBrewer replaced by importFrom of functions \code{boot} and \code{brewer.pal}.
    }
  }
}


\section{Changes in version 2.2-2 [2021-06-03]}{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqidegrad()} degradation index (replaces \code{seqprecorr()} that
      becomes obsolete). Class \code{seqprecorr} now named \code{seqidegrad} and associated print method renamed accordingly.
      \item
      \code{seqibad()} badness index.
      \item
      \code{seqinsecurity()} insecurity index.
      \item
      \code{dissdomassoc()} computes measures proposed in Piccarreta (SMR 2017) to assess the degree of association between domains.
    }
  }


  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqdist()} can now compute distances between two subsets of sequences.
      The subsets are defined by providing a list with two subsets of sequence
      indexes as \code{refseq} argument.
      \item
      \code{seqdist()} dimnames of the objects returned by seqdist are now all set
      from the rownames of the provided sequence object.
      \item
      \code{seqdist()} \code{method="DHD"} now supports \code{sm="INDELS"} and \code{sm="INDELSLOG"}.
      \item
      \code{seqcost()} \code{time.varying=TRUE} works with \code{"INDELS"} and \code{"INDELSLOG"}.
      \item
      \code{seqdistmc()} gains a \code{what} argument to choose what value will
      be returned: distances, costs, or combined state sequences.
      \item
      \code{seqintegration()} renamed \code{seqintegr()}
      \item
      \code{seqindic()} the possible \code{indic} list gains elements
      \code{'degrad'} for degradation index,
      \code{'bad'} for badness index,
      \code{'insec'} for insecurity index,
      and the type 2 forms
      \code{'meand2'}, \code{'dustd2'}, \code{'turb2'}, and \code{'turb2n'}
      taking non-visited states into account.
      In addition, \code{indic} accepts now also
      value \code{'ranked'} that is equivalent to \code{c('degrad','bad','prec','insec')}.
      \item
      \code{seqindic()} value \code{'inpos'} of \code{indic} renamed \code{'integr'}, \code{'volat'} now in the \code{'complexity'} group.
      \item
      \code{seqindic()} changes in order of indicators in the returned table.
      \item
      \code{seqformat()} with \code{from='SPELL'} and \code{process=TRUE} now issues
      a warning message when the max of the \code{end} column exceeds \code{limit}.
      \item
      \code{seqlegend()} gains an argument \code{boxes} to control how colors are displayed. In addition, now supports any argument of function \code{legend} and, e.g. \code{x} (or \code{x} and \code{y}) can be used instead of \code{position}.
      \item
      \code{seqstatf()} gains the \code{with.missing} argument.
      \item
      \code{seqlogp()} gains the \code{with.missing} argument. In addition,
      the overall state distribution can now be used as starting point by
      setting \code{begin='global.freq'.}
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqlogp()} error for state sequences with missing values.
    }
  }

  \subsection{Misc.}{
    \itemize{
      \item
      New imports from psych and cNORM (for \code{dissdomassoc})
    }
  }


}


\section{Changes in version 2.2-1 [2020-10-24]}{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqivardur()} computes the variance of spell durations for each individual sequence.
      Either the variance of the observed spells or by considering also the 0-time spent
      in non-visited states. The associated print method can print
      the mean, standard deviation, variance, and max variance.
      %\item
      %\code{seqidegrad()} degrading index.
      %\item
      %\code{seqibad()} badness index.
    }
  }

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqST()} gains an argument \code{type} to select the type of duration variance used.
      \item
      \code{seqindic()} the possible indicator list gains elements
      \code{'recu'} for recurrence index,
      \code{'nvolat'} for normative volatility,
      %\code{'degrad'} for degrading index,
      %\code{'bad'} for badness index,
      %\code{'prec2'} for revised precarity index,
      \code{'meand'} for mean spell duration,
      \code{'dustd'} for standard deviation of the duration.
      %, as well as the type 2 forms \code{'meand2'}, \code{'dustd2'},
      %\code{'turb2'}, and \code{'turb2n'}.
      The \code{indic} argument accepts now also values \code{'basic'}, \code{'diversity'},  \code{'complexity'}, and \code{'binary'} that each selects a subset of indicators.
      \item
      \code{seqprecstart()} the normalized \code{stprec} has now a minimum 0.
      \item
      \code{seqdist()} the check of allowed number of unique sequences now
      depends on \code{refseq} and used thresholds are \code{sqrt(.Machine$integer.max)} when
      \code{refseq==NULL}, and \code{.Machine$integer.max} otherwise.
      In addition, a new argument permits to skip the test.
    }
  }

  \subsection{Bug Fixes}{
    \itemize{
      \item
      \code{seqdur()} returned sometimes bad values for the duration of the
      last spell of sequences of unequal lengths when \code{with.missing=TRUE}.
      \item
      \code{seqdur()} unexpected outcome for sequences with missing values when \code{with.missing=FALSE}. Now correctly ignores missing values.
      \item
      \code{seqdplot()} and \code{plot.stslist.statd()} ignored the \code{xlab} argument.
      \item
      \code{seqconc()}, \code{seqdecomp()}, \code{seqformat()}, and the print method for
      \code{stslist} objects produced in some circumstances output without the row names of the input data.
      \item
      \code{seqsubsn()} error for sequences with missing values when \code{with.missing=FALSE}. Now correctly ignores missing values.

    }
  }

  \subsection{Misc.}{
    \itemize{
      \item
      TraMineRextras added to the 'Suggests' list (to avoid NOTE on undeclared package in Rd xrefs.)
    }
  }
}

\section{Changes in version 2.2-0.1 [2020-04-29]}{

  \subsection{Misc.}{
    \itemize{
      \item
      \code{seqdef()}: switched back from [grDevices] \code{hcl.colors} to [colorspace] \code{qualitative_hcl} to avoid dependence on R >=3.6 necessary for \code{hcl.colors}.
    }
  }
}



\section{Changes in version 2.2-0 [2020-04-22]}{

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqdist()}, argument \code{indel} can now be \code{"auto"} (new default)
      in which case a suitable indel value is generated for the provided \code{sm}.
      \item
      \code{seqdist()} now uses the \code{rbind} stslist method to add \code{refseq} to
      the set of sequences when it is a state sequence object.
      (Transparent to the user.)
      \item
      \code{seqdist()} now gives a zero weight to the \code{refseq} when specified
      as a state sequence object.
      \item
      \code{seqdist()} \code{refseq} including \code{refseq} as a state sequence
      object can be used with all methods (did not work with OMstran, CHI2, and EUCLID).
      \item
      \code{seqdist()} vector \code{indel} now supported by OMspell and OMslen (in addition to OM and OMstran).
      \item
      \code{seqdist()}, normalization now also possible for OMspell, OMslen, OMstran, OMloc, and TWED.
      \item
      \code{seqdef()} now uses \code{hcl.colors} (grDevices) instead of
      \code{qualitative_hcl} (colorspace) to set default colors when n>13.
      (Transparent to the user.)
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{rbind.stslist()} bad handling of missing states.
      \item
      \code{seqdist()}, argument \code{sm} did not recognize values \code{"INDELS"} and
      \code{"INDELSLOG"}.
      \item
      \code{seqdist()} raised warnings when \code{method='OMslen'} because of a
      vector of conditions in an if test.
      \item
      \code{seqdist()} ignored the \code{weighted} value when computing substitution
      costs with the sm method.
      \item
      \code{seqdist()} for OMstran ignored the \code{weighted} value when computing
      probability-based indels (i.e. with \code{transindel="prob"}).
      \item
      \code{seqdist()} TWED wrongly depended on the indel value. Now indel
      serves only in the improbable case of empty sequences.
      \item
      \code{seqdist()} \code{maxlength} and \code{gmean} normalization generated negative
      distances in some circumstances. Fixed by setting the maximum length as
      max length x max indel.

   }
  }
  \subsection{Misc.}{
    \itemize{
      \item
      Deleted unused OMVI2distance h and cpp files.
    }
  }
}


\section{Changes in version 2.0-15 [2020-03-18]}{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqintegration()} index of integrative potential of C. Brzinsky-Fay.
      \item
      \code{seqivolatility()} index of objective volatility of C. Brzinsky-Fay.
      \item
      \code{print.seqipos()} print method for objects returned by \code{seqipos()}.
      \item
      \code{is.stslist()} checks that the object is a proper \code{stslist} object.
      \item
      \code{rbind.stslist()} \code{rbind} method for state sequence objects.
    }
  }

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqdef()} now provides a default color palette (using colorspace) when number of states exceeds 12.
      \item
      \code{alphabet()} gains an argument \code{with.missing}.
      \item
      \code{seqipos()} gains an argument \code{index} to select between three types of indicators: share of positive states, volatility, or integrative potential for the positive state.
      \item
      \code{seqipos()} returned object has an attribute \code{sbinary} with the derived binary sequences.
      \item
      \code{seqindic()} possible values for argument \code{indic} include now the indicators of objective volatility and integrative potential both for the provided sequences and the positive-negative sequences derived from them.
      \item
      \code{seqdist()}: OMloc, OMslen, OMspell, and OMstran now also apply to sequences of different lengths.
      \item
      \code{seqdist()}: \code{with.missing=TRUE} now supported by all methods.
      \item
      \code{seqdist()} raises a warning when there are empty sequences except for OMloc where an error is raised.
      \item
      \code{seqcost()}: \code{with.missing} forced as \code{FALSE} (as in \code{seqdist}) when there are no non-void missings in the sequences.
      \item
      \code{seqprecorr()}: new methods \code{"FREQ+"}, \code{"TRATE+"}, \code{"TRATEDSS+"}, \code{"RANK+"}     that adjust the returned penalty for the mean transition weight.
      \item
      \code{seqprecorr()}, a same state value is assigned to equivalent states when counting transitions and computing transition probabilities (methods \code{FREQ}, \code{TRATE}, \code{TRATEDSS}).
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      Print methods for \code{seqprec} and \code{seqprecorr} objects did not work.
      \item
      \code{print.stslist()} did not print ending missing values when \code{format='SPS'}.
      \item
      \code{print.stslist()}, when \code{format='SPS'}, additional parameters for controlling the format were ignored.
      \item
      \code{seqformat()} did not account for the \code{right} argument when \code{to='SPS'}.
      \item
      \code{seqdist()} with \code{method=CHI2}. Fixed issue with unexpected warnings when over an interval all sequences stay in the same state.
      \item
      \code{seqdist()} raised an error for some methods when the unused argument \code{indel} was a vector.
      \item
      \code{seqdist()}, argument \code{sm}: values \code{'CONSTANT'} and \code{'TRATE'} were only accepted for \code{method='OM'} and \code{'HAM'}. Now can be used with all methods that require the \code{sm} argument.
      \item
      \code{seqplot()} raised an error when deprecated \code{tlim} argument was used. Now a warning.
      \item
      \code{[]} subsetting of \code{stslist} objects with negative column
      indexes produced unexpected results.
    }
  }
}

\section{Changes in version 2.0-14 [2020-01-13]}{

  \subsection{Bug Fixes:}{
    \itemize{
      \item
      \code{print.stslist.freq()} wrongly used \code{width} argument.
    }
  }
}

\section{Changes in version 2.0-13 [2019-11-19]}{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqindic()} returns a table with per sequence values of selected indicators.
      \item
      \code{seqipos()} proportion of positive spells/states per sequence.
      \item
      \code{seqprecarity()} precarity index (previously in TraMineRextras)
      \item
      \code{seqprecorr()} correction factor for precarity index (previously in TraMineRextras)
      \item
      \code{seqprecstart()} state precarity level for precarity index (previously in TraMineRextras)
    }
  }

  \subsection{Changes in existing functions:}{
    \itemize{
      \item
      \code{seqlength()}, \code{seqST()}, and \code{seqsubsn()} gain a \code{with.missing} argument to control how missing values should be treated.
    }
  }

  \subsection{Bug Fixes:}{
    \itemize{
      \item
      \code{seqST()} normalization did not account for non-void missing values when present.
    }
  }

  \subsection{Misc.:}{
    \itemize{
      \item
      Fixed issue with \code{\\traminer} in vignette.
    }
  }
}



\section{Changes in version 2.0-12 [2019-06-21]}{

  \subsection{Changes in existing functions:}{
     \itemize{
       \item
       \code{seqdef()} now raises an error when argument \code{alphabet} contains duplicate elements.
       \item
       \code{alphabet()} in its assign form now raises an error
        when the assigned vector contains duplicate elements.
       \item
       \code{seqdist()} gains a new argument \code{global.pdotj}
       to control the marginal distribution to be used for the \code{'CHI2'} distance.
       \item
       \code{seqdist()}, \code{'CHI2'} and \code{'EUCLID'} distances are now, when
       \code{norm='auto'}, normalized by the maximal distance, which
       for \code{'CHI2'} depends on the marginal state distribution.
       \item
       \code{seqici()}, \code{seqient()}, and \code{seqST()} gain a new argument \code{silent} to control whether messages about running operations should be displayed.
       \item
       \code{seqST()}: suppression of useless messages when \code{norm=TRUE}.
     }
  }

  \subsection{Misc.:}{
    \itemize{
      \item
      Suppressed unused argument \code{weighted} from internal \code{OMstran} function.
      \item
      Help page for \code{seqdist}: description of the \code{weighted} argument adjusted to reflect that it applies only to the \code{"CHI2"} distance.
      \item
      Updated links to Swiss Household Panel in \code{actcal} and \code{biofam} help pages.
    }
  }
}

\section{Changes in version 2.0-11.1 [2019-03-22]}{

  \subsection{Bug Fixes:}{
    \itemize{
      \item
      eventseq.cpp: rchk message 'calling allocating function \code{Rf_asChar} with argument allocated using \code{TMRNumberFormat}'.
    }
  }
}

\section{Changes in version 2.0-11 [2019-03-18]}{

  \subsection{Changes in existing functions:}{
     \itemize{
       \item
       \code{seqdef()}. Speed improvement suggested by Jouni Helske. Transparent for the user.
       \item
       \code{seqdist()}. An error is raised when the number of unique sequences exceeds the maximal allowed.
       \item
       \code{plot.ststlist.statd()}. Now checks the \code{type} argument and returns the values plotted.
     }
  }
  \subsection{Bug Fixes:}{
    \itemize{
      \item
      \code{seqtab()}, \code{seqxtract(), checkcost(), SPELL_to_STS()}. Issues with \code{length > 1 in coercion to logical}.
      \item
      Warning about potential stack imbalanced \code{PROTECT} in relation with \code{TMRNumberFormat} in the C++ code (eventseq.cpp).
    }
  }

}


\section{Changes in version 2.0-10 [2018-11-18]}{

  \subsection{Bug Fixes:}{
    \itemize{
      \item
      \code{seqdist()}: when \code{refseq} was passed as a state sequence
      object wrong results were sometimes returned.
      \item
      \code{seqdist()}: CHI2 and EUCLID distances were computed using
      counts instead of proportions.
      \item
      \code{seqdist()}: CHI2 and EUCLID bad behavior in presence of missings.
      \item
      Vignette: Fixed issue with updated \code{fancyvrb.sty}.
    }
  }
  \subsection{Misc.:}{
     \itemize{
       \item
       Vignette: Error in formula for complexity index.
       Also, now DOI instead of link to JSS article.
     }
  }
}

\section{Changes in version 2.0-9 [2018-08-20]}{

  \subsection{Changes in existing functions:}{
    \itemize{
      \item
      \code{seqplot()} now supports the \code{ncol} argument for controlling the number of columns in the color legend.
      \item
      \code{seqdef()}: new argument \code{tick.last} to set the \code{tick.last} attribute of the state sequence object. Default is \code{tick.last = FALSE} to preserve the previous behavior.
      \item
      \code{plot.stslist()}, \code{plot.stslist.freq()}, \code{plot.stslist.statd()}, \code{plot.stslist.modst()}, \code{plot.stslist.rep()} (i.e. also \code{seqplot} with type \code{"i"}, \code{"I"}, \code{"f"}, \code{"d"}, \code{"Ht"}, \code{"ms"}, or \code{"r"}), and \code{plot.seqdiff()}: new argument \code{tick.last} that when set as \code{TRUE} allows to enforce a tick mark at the last position on the time x-axis. Has no effect when the last position is 1 + a multiple of \code{xtstep}. Default is to use the \code{tick.last} attribute of the state sequence object.
      \item
      \code{seqformat()}:
      \itemize{
        \item
        For \code{from = "SPELL"} and \code{process = FALSE}, \code{pdata = "auto"} is now equivalent to \code{NULL} instead of raising an error. Also, using argument \code{pvar} with \code{pdata = NULL} now raises a warning instead of an error and \code{pvar} is simply ignored.
        \item
        For \code{from = "SPELL"} an error is raised when columns referred to
        by \code{begin} and \code{end} are of class "Date".
        Integer values are expected.
        \item
        When \code{from = "SPELL"} or \code{to = "SPELL"}, an error is raised
        when the birth year column of \code{pdata} is of class "Date".
        Integer values are expected.
      }
      \item
      \code{seqplot()}: for \code{type = "r"} (\code{seqrplot}), an error is raised when
      a group has less than two cases.

    }
  }
  \subsection{Bug fixes:}{
    \itemize{
      \item
      \code{seqformat()}: An unneeded warning was raised when data
        was a matrix with a single string element.

      \item
      \code{seqdef()}: Bad handling of \code{missing} argument when
      \code{informat = "SPS"} or \code{"SPELL"}.

    }
  }
}

\section{Changes in version 2.0-8 [2018-01-26]}{

  \subsection{Changes in existing functions:}{
    \itemize{
      \item
      \code{seqcost()}: argument \code{miss.cost.fixed} is now \code{NULL} by
      default and will be set as \code{FALSE} when \code{method = "INDELS"} or
      \code{"INDELSLOG"}, and as \code{TRUE} otherwise.
      \item
      \code{seqpcplot()}: new logical \code{weighted} argument to control
      whether weights should be used or not.
    }
  }

  \subsection{Bug fixes:}{
    \itemize{
      \item
      \code{seqrplot()} and \code{seqplot(..., type="r", ...)}: the \code{method}
      and other related arguments for computing the \code{diss} matrix when the
      later was not provided was not recognized.
      \item
      \code{seqplot()}: Fixed some unused argument issues.
    }
  }

  \subsection{Misc:}{
    \itemize{
      \item
      Vignette: Suppressed the loading of two unused LaTeX packages
      (subfigure and afterpage) that prevented the vignette to be built
      under OSX.
      \item
      Declaring use of C++11 for log1p: \code{SystemRequirements: C++11} declaration
      in \code{DESCRIPTION} and \code{math.h} header in \code{NMSdistance.cpp}
      \item
      Internal function \code{checkargs()} renamed as \code{TraMineR.check.depr.args()}
      and made public for use in \code{TraMineRextras}.
    }
  }
}


\section{Changes in version 2.0-7 [2017-08-15]}{
  \subsection{Changes in existing functions:}{
    \itemize{
      \item
      \code{alphabet()}: the get form now also applies to event sequences.
      In addition, an error is now raised when the argument is not a state sequence object, an event sequence object, or a probabilistic suffix tree (see the PST package for the latter).
      \item
      \code{seqdecomp()}: the \code{miss} argument can now also be a vector, e.g. \code{miss = c("*", "\%")}
      \item
      \code{seqformat()} has a new \code{right} argument to be used with
      \code{to = 'SPELL'}. The default \code{right = 'DEL'} suppresses the end spells of missing values. Set \code{right = NA} to keep the end spells of missing values.
      \item
      \code{seqdef()} now raises an error message when \code{void} is not a character different from \code{left}, \code{gaps}, and \code{right}.
      \item
      \code{seqtrate()} gains a new argument \code{count}. When \code{count = TRUE}, the function returns counts of transitions instead of transition probabilities.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqpcplot()}:
      Due to a change in the R \code{tapply} function, option \code{ltype="non-embeddable"} did no longer work.
      \item
      \code{seqefsub()} with non-null \code{str.subseq} argument:
       multiple partial matches between events in \code{str.subseq} and the alphabet of events of
      the \code{eseq} event sequence object crashed the R session. Fixed by a change in the internal \code{seqecreatesub} function.
      \item
      \code{seqformat()}: \code{to="TSE"} and \code{to="SPELL"} produced errors or unexpected results in presence of missing and/or void states. Now, when \code{to = "SPELL"}, the missing and void codes are both converted to \code{NA} before conversion while they are kept as is when \code{to = "TSE"}.
      \item
      \code{seqetm()} produced an error in presence of void elements.
      \item
      An error occurred in the internal \code{implicativestat} function when a same condition subsequence was present in all subsequences.
    }
  }
  \subsection{Misc:}{
    \itemize{
      \item
      Documentation of \code{seqformat()} and \code{seqecreate()}: added examples of the handling of missings in conversion into TSE and SPELL format.
    }
  }
}

\section{Changes in version 2.0-6 [2017-06-17]}{
  \subsection{Changes in existing functions:}{
    \itemize{
      \item
      \code{seqformat()}:
      \itemize{
        \item
        Redesigned version of the function with a new extensible, robust and documented
        implementation.
        \item
        Rewritten documentation with argument types, default values, scopes and
        detailed explanations.
        \item
        Added and generalized conversion to \code{"SPELL"} from TraMineRextras;
        added a \code{with.missing} argument specifically for this conversion.
        \item
        Added the possibility to pass directly the unique individual IDs (row names)
        of the input sequences with \code{id} when converting to \code{"TSE"}.
        \item
        Clarified the meaning of the \code{id} argument and changed its position in the
        list of arguments to reflect this.
        \item
        Changed the default value of the \code{id}, \code{begin}, \code{end}, and
        \code{status} arguments (now \code{"SPELL"} oriented).
        \item
        Renamed \code{compressed} argument as \code{compress} to avoid
        confusion (here it applies to the output, not to input data).
        \item
        Renamed \code{nr} argument as \code{missing} to avoid confusion (here
        it specifies the code to consider as missing values in input data,
        which is not the TraMineR internal code for missing values in state
        sequence objects).
        \item
        Adapted function calls to match the renamed argument names
        (\code{compressed}, \code{nr}) and the new input type checking (ie.
        single strings are deprecated).
        \item
        See the updated documentation of \code{seqformat()} for details.
      }
      \item
      \code{SPS_to_STS()}: renamed \code{nr} argument as \code{missing} to match
      \code{seqformat()} argument names renaming.
    }
  }
  \subsection{Bug fixes:}{
    \itemize{
      \item
      \code{seqformat()}:
      \itemize{
        \item
        From \code{"SPELL"} to \code{"TSE"}: output IDs were incorrect.
        \item
        From \code{"SPELL"} to \code{"STS"}: states not appearing in the
        data were dropped from the output column levels.
      }
    }
  }
  \subsection{Misc:}{
    \itemize{
      \item
      Added new internal helper functions:
      \itemize{
        \item
        \code{msg.warn0()}: same as the existing \code{msg.warn()} but without
        white space insertion.
        \item
        \code{is.positive.integers()}: check if an object is a vector of
        positive integers.
        \item
        \code{is.a.character()}: check if an object is a (unique) character.
        \item
        \code{is.a.string()}: check if an object is a string.
        \item
        \code{is.strings()}: check if an object is a vector of strings.
        \item
        \code{is.index()}: check if an object is a positive integer or a string.
        \item
        \code{is.indexes()}: check if an object is a vector of positive integers
        or strings.
        \item
        \code{checkindex()}: check if an object is a valid data frame or matrix
        index; otherwise, an error is raised and an information message is
        displayed.
        \item
        \code{checkindexes()}: check if an object is a vector of valid data
        frame or matrix indexes; otherwise, an error is raised and an information
        message is displayed.
      }
      \item
      Grouped internal \code{is.xxx} helper functions into a single
      TraMineR-is_helpers.R file.
      \item
      Generalized \code{msg.warn()} / \code{msg.warn0()} and \code{msg()} /
      \code{msg0()} code.
      \item
      Internal C/C++ function \code{tmrWeightedInertiaDist()} made public as the R function \code{TraMineRInternalWeightedInertiaDist()}. (Used by package \code{Weightedcluster})
    }
  }
}


\section{Changes in version 2.0-5 [2017-05-13]}{

\subsection{Note:}{
  \itemize{
    \item
    This is a major update of the CRAN version of TraMineR.
    \item
    Check also changes in versions 1.9-14, 2.0-0, 2.0-1, 2.0-2, 2.0-3, and 2.0-4 that have not been released on the CRAN.
    }
}

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqtrate()}: now accepts a sequence object (\code{seqdata} argument) containing only one sequence.
    \item
    \code{seqdist()}: now accepts a reference sequence object (\code{refseq} argument)
    containing missing values while the main sequence object (\code{seqdata} argument) doesn't.
    }
}

\subsection{Misc:}{
  \itemize{
    \item
    Replaced default deprecated values of \code{norm} in
    \code{seqdistmc()} and \code{seqtree()}.

    \item
    Fixed issues with examples in documentation page of
    \code{seqtree()}.
    }
}
}
\section{Changes in unreleased version 2.0-4 [2017-04-13]}{


\subsection{Changes in existing functions:}{
  \itemize{
    \item
    Renamed several argument names to increase consistency within TraMineR and
    between TraMineR and R. The aim is also to have a common naming
    convention within TraMineR.
    A new internal function (\code{checkargs()})is used to guarantee
    backward compatibility. If the old argument name is used instead of the new
    one, a warning message with an explanation is displayed and the execution
    continues. If the new and old argument names are used together, an error
    message is displayed and the execution stops.
    The following functions have at least one renamed argument:
    \code{dissrep()}, \code{disstree()}, \code{disstree2dot()},
    \code{disstree2dotp()}, \code{disstreedisplay()}, \code{is.eseq()}, \code{is.seqelist()}, \code{seqdiff()},
    \code{seqeconstraint()}, \code{seqecontain()}, \code{seqecreate()}, \code{seqefsub()}, \code{seqeid()},
    \code{seqelength()}, \code{seqelength<-}, \code{seqetm()}, \code{seqeweight()}, \code{seqeweight<-},
    \code{seqlegend()}, \code{seqpcplot()}, \code{seqplot()}, \code{seqrep()}, \code{seqtab()}, \code{seqtrate()},
    \code{seqtree()}, \code{seqtree2dot()}, \code{seqtreedisplay()}, \code{seqeisweighted()},
    \code{plot.seqalign()}, \code{plot.seqdiff()}, \code{plot.stslist()}, \code{plot.stslist.freq()},
    \code{plot.stslist.meant()}, \code{plot.stslist.modst()}, \code{plot.stslist.rep()},
    \code{plot.stslist.statd()}, \code{plot.subseqelistchisq()}. \cr
    See the help page of each of
    these functions for the mapping between old and new argument names.

    \item
    Renamed function: \code{is.seqe()} was renamed as \code{is.eseq()}.
    }
}

\subsection{Misc:}{
  \itemize{
    \item
    Fixed an issue with an example in the documentation page of \code{seqpcplot()}.
    \item
    Fixed equation typing errors in \code{seqrep()} documentation.
    }
}
}

%% =========
\section{Version 2.0-3 [2017-04-06] (Unreleased version)}{

\subsection{Misc:}{
  \itemize{
    \item
    In \code{src/tmrsequence.cpp} from TraMineR 1.8-13:
    \itemize{
      \item
      Fixed two memory errors detected by Valgrind.
      \item
      Fixed a PROTECT error.
      }
    \item
    Fixed issues with examples in documentation pages of
    \code{dissmfacw}, \code{disstree}, \code{seqtree}, and \code{plot.stslist.meant}.
    }
}}

\section{Version 2.0-2 [2017-03-28] (Unreleased version)}{

\subsection{Misc:}{
  \itemize{
    \item
    Changed character encoding from latin1 (ISO-8859-1) to UTF-8.
    \item
    Normalized line endings: LF.
    }
}
}

\section{Version 2.0-1 [2017-03-27] (Unreleased version)}{

\subsection{Misc:}{
  \itemize{
    \item
    Removed the following unused functions: \code{vidx()}, \code{seqmatsaltt()}, \code{seqmathenikoff()}.
    \item
    Removed the functions deprecated in TraMineR 1.x: \code{seqesetlength()}, \code{dissreg()}, \code{dissmfac()}.
    }
}
}

\section{Version 2.0-0 [2017-03-27] (Unreleased version)}{

\subsection{New function:}{
  \itemize{
    \item
    \code{seqcost()}: Evolution of \code{seqsubm()} that offers different ways (CONSTANT,
    TRATE, INDELS, INDELSLOG, FUTURE, FEATURES) to determine indel and
    substitution costs (see the documentation of \code{seqcost()} for details). Unlike
    \code{seqsubm()}, \code{seqcost()} returns both the indel and the substitution costs.
    }
}


\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqdist()}:
      \itemize{
        \item
        New major version with many new features and a new extensible, robust
        and documented implementation (R code).
        \item
        New methods: localized OM (\code{OMloc}), spell length sensitive OM (\code{OMslen}),
        OM of spells (\code{OMspell}), OM of sequences of transitions (\code{OMstran}),
        Time Warp Edit Distance (\code{TWED}), Number of Matching Subsequences (\code{NMS}),
        Number of Matching Subsequences weighted by the Minimum Shared Time (\code{NMSMST}),
        Subsequence Vectorial Representation (\code{SVRspell}), Euclidean distance (\code{EUCLID}),
        Chi-squared distance (\code{CHI2}).

        \item
        New arguments: \code{kweights}, \code{tpow}, \code{expcost}, \code{context}, \code{link}, \code{h}, \code{nu},
        \code{transindel}, \code{otto}, \code{previous}, \code{add.column}, \code{breaks}, \code{step}, \code{overlap},
        \code{weighted}, \code{prox}.

        \item
        \code{sm}: value \code{"CONSTANT"} has been removed for DHD as it doesn't make sense
        and the values \code{"INDELS"} and \code{"INDELSLOG"} have been added (see \code{seqcost()} documentation).

        \item
        \code{norm}: value \code{TRUE} is replaced by \code{"auto"} and \code{FALSE} by \code{"none"}.

        \item
        See the updated documentation of \code{seqdist()} for details.
        }
    \item
    \code{seqsubm()}: This is now an alias for \code{seqcost(...)$sm}.
    }
}
}

\section{Version 1.8-14 [2017-03-20] (Unreleased version)}{

\subsection{Misc:}{
  \itemize{
    \item
      Registration of native routines for R 3.4.x.
      See \url{https://stat.ethz.ch/pipermail/r-devel/2017-February/073755.html}.
      }
}
}

\section{Version 1.8-13}{

\subsection{Misc:}{
  \itemize{
    \item
    Changes in C-code for the \code{seqefsub} function: replaced call
    to function round by a call to \code{fround} to comply with
    forthcoming changes in R 3.4.0 (request of Brian D. Ripley).
    The change is transparent for the user.
    }
}
}

\section{Version 1.8-12}{

\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqST()}: new argument \code{norm} to ask for a normalized
    turbulence index.
    \item
    \code{seqformat()}: The transformation now stops with an error message
    when the columns referenced with the begin and end
    argument contain a non integer value.
    }
}

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{plot.stslist()}: an unnecessary warning occurred when a
    vector of labels was passed as \code{ytlab} argument.
    \item
    \code{seqdef()}: now accepts to create a state sequence object
    with an alphabet that has only one element. Fixed an error
    that occurred when there was only one state
    }
}

\subsection{New data examples:}{
\itemize{
    \item
    \code{bfspell}: a small data set with 20 sequences in SPELL format.
    }
}

\subsection{Misc:}{
  \itemize{
    \item
    updated \code{seqformat} help page: now includes an example of a
    transformation from SPELL to STS.
    \item
    fixed bad use of \code{extern "C" {}} in TraMineR.h
    (done by B. Ripley, CRAN version 1.8-11.1)
    }
}
}

\section{Version 1.8-11}{

\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqmeant()}: New \code{serr} argument. When \code{serr=TRUE},
    \code{seqmeant} computes
    the variance and standard deviation of the total times spent in the
    different states, and the standard error of the mean total times.
    \item
    \code{seqmtplot()}: When \code{serr=TRUE}, error bars are displayed in the
    mean time plot.
    \item
    \code{seqdist()}:
      \itemize{
        \item
        New error message when \code{sm=NA} with method \code{"OM"}.
        \item
        New error message when \code{refseq} is a state sequence object with
        an alphabet assigned to it different from that of \code{seqdata}.
        }
    }
}

\subsection{Misc:}{
  \itemize{
    \item
    updated \code{disstreedisplay} help page (\code{tree} argument).
    \item
    updated \code{seqdist} help page (\code{refseq} argument and example).
    \item
    updated CITATION file (new ref and fixed a doi argument).
  }
}
}

\section{Version 1.8-10}{

\subsection{Misc:}{
  \itemize{
    \item
    Updated help pages: \code{seqeconstraint}, \code{seqtree}, \code{distree2dot}.
    \item
    Added required basic packages to the import statements to comply with R v3.3 requirements.
    \item
    One additional exported alias to a TraMineR internal function:
    \code{TraMineRInternalSeqgbar}.
    }
}
}

\section{Version 1.8-9}{

\subsection{New function:}{
  \itemize{
    \item
    \code{seqpcfilter()}: convenience function to define the coloring
    filter options to be passed as \code{filter} argument to \code{seqpcplot()}.
    }
}

\subsection{Changes in existing functions:}{
\itemize{
  \item
  \code{seqpm()}: New \code{sep} argument to allow searching for string patterns when
    states are not labelled with single characters.
  \item
  \code{seqpcplot()}:
  \itemize{
    \item
    New argument \code{seed} to control the jittering.
    \item
    The \code{filter} argument can now simply be a scalar, in which case
    the \code{minfreq} filter is applied with this numeric value as threshold.
    See also the new function \code{seqpcfilter()}.
    \item
    New argument \code{missing} to control  whether and how to display missing
    values.
    }
  }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqformat()}: fixed error occurring when converting from STS to TSE
    with a \code{tevent} matrix containing empty strings (i.e. \code{""})
    \item
    \code{dissmfacw()}: reported F values now obtained by dividing the within
    discrepancy in the denominator by (\eqn{n-m}), where \eqn{n} is the sample size
    and \eqn{m} the total number of predictors (contrasts for categorical
    factors). Up to here (\eqn{n-m-1}) was mistakenly used
    (Reported by Vicente Ponsoda.)
    }
}
\subsection{Misc:}{
  \itemize{
    \item
    Two additional exported aliases to TraMineR internal functions:
    \code{TraMineRInternalSeqeage} and \code{TraMineRInternalLegend}.
    }
}
}

\section{Version 1.8-8}{

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqecreate()}: an error is thrown when events are not grouped
    by id in inputted TSE data. (Reported by Nicolas Jay).
    This requirement is now specified in \code{seqecreate} help page.
    }
}
\subsection{Misc:}{
  \itemize{
    \item
    exported alias functions allowing other packages to access TraMineR
    internal functions (see \code{?TraMineRInternalLayout}).
    }
}
}

\section{Version 1.8-7}{

\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqpcplot()}: Suppressed unnecessary \code{output} argument. The
    \code{seqpcplot} object is automatically retrieved when using the
    assignment operator, e.g., \code{p <- seqepcplot(...)}.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqformat()}: When converting from STS to TSE, an error was raised if the
    tevent matrix had empty strings (i.e. \code{""}). Now, this is considered as no event.
    \item
    \code{seqpcplot()}: Fixed error that appeared at the use of \code{"_end"} events.
    \item
    \code{seqpcplot()}: An error occurred when plotting a state sequence object
    (of class \code{stslist}) with a numeric \code{cnames} attribute.
    }
}
\subsection{Misc:}{
  \itemize{
    \item
    Required packages \code{RColorBrewer} and \code{boot} now listed as \code{"imports"}
    in DESCRIPTION and using \code{import} in NAMESPACE.
    }
}
}

\section{Version 1.8-6}{

\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqformat()}: new \code{nr} argument to specify the missing state
    symbol in SPS input.
    \item
    \code{disstreeleaf()}: new logical \code{label} argument to specify
    whether the leaf membership should be labelled with
    the classification rules.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqtreedisplay()} and \code{disstreedisplay()}: GraphViz installer no longer
    adds GraphViz to the PATH environment variable. Therefore the two
    functions have been adapted to search for GraphViz. In case GraphViz
    would not be found, you can specify the GraphViz installation
    directory with the new \code{gvpath} argument.
    \item
    \code{disstree()} and \code{seqtree()}: removed the warning when \code{R} equals \code{0} or \code{1}  (no permutations).
    \item
    \code{seqformat()}: fixed a problem with missing states when converting
    from SPS to STS (see Changes in function above.)
    \item
    \code{seqpcplot()}: fixed issues with \code{which} argument of \code{plot.seqpcplot}.
    \item
    \code{seqpcplot()}: fixed issues with arguments \code{xlab} and \code{title}.
    \item
    \code{seqpcplot()}: replaced a warning message by an error message at
    failures in finding plot positions for sequences. The error
    message advices to modify the (currently hidden) \code{maxit} argument.
    Additionally, automatically generated subtitles are now hidden
    when the argument \code{title} is used.
    \item
    \code{as.character.seqelist()}, \code{print.seqelist()}: fixed an issue with time
    display in event sequences which was in scientific notation for numbers
    with more than 2 digits. The function now uses the R format function and
    thus accounts for global formatting options such as \code{options(digits=)}
    and/or \code{options(scipen=)}.
    \item
    \code{seqtrate()}: fixed error with sequence objects having only two columns.
    }
}
\subsection{New functionalities:}{
  \itemize{
    \item
    \code{disstree2dot()} and \code{disstree2dotp()} gain a new argument called \code{title.outer}.
    If \code{title.outer=TRUE}, the title is printed in the outer margins.
    }
}
}

\section{Version 1.8-5}{

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqefsub()}: reported support did not properly account for weights.
    \item
    \code{seqtreedisplay()}: corrected a bug when using representative sequences and
    a \code{dist} object was passed to the \code{dist.matrix} argument (reported by
    Emanuela Struffolino).
    \item
    \code{seqLLCS()} and \code{seqLLCP()}: added a check on the argument. Both sequences
    should belong to state sequence objects with a common alphabet.
    \item
    \code{seqpcplot()}: small change in default lower \code{ylim}.
    }
}
\subsection{Help pages:}{
  \itemize{
    \item
    help pages updated with author and keyword fields.
    }
}
}

\section{Version 1.8-4}{

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqrep()}, \code{dissrep()}, \code{seqrplot()}: wrong (unweighted) \code{"na"} values were returned;
    also quality measures \code{"MD"} (mean distance to representative) and \code{"V"}
    (discrepancy) were not computed properly when more than one representative
    selected (since version 1.8-2).
    }
}
\subsection{User invisible changes:}{
  \itemize{
    \item
    Added an internal function to fix an issue with an internal C level
    function when called from other packages.
    }
}
}

\section{Version 1.8-3}{

\subsection{Information pages:}{
  \itemize{
    \item
    Updated online help pages.
    \item
    Updated list of references returned by \code{citation("TraMineR")}.
    }
}
\subsection{New functionalities:}{
  \itemize{
    \item
    \code{seqpcplot()}: parallel coordinate plot for sequence data.
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqdss()}: adding long state labels to returned sequence object.
    \item
    \code{seqplot()}: new option \code{type = "pc"}.
    \item
    \code{seqplot()}, \code{seqdplot()}, \code{seqiplot()}, \code{seqfplot()}, \code{seqmsplot()}, \code{seqrplot()}:
    if \code{density} and/or \code{angle} are used to produce shading lines
    instead of solid colors, the legend is plotted using the same
    parameters and thus corresponds to the colors/shades used in the plot.
    \item
    \code{seqplot()}: if \code{group} argument is a factor, the plots are
    now ordered the same way as the factor levels.
    \item
    \code{seqplot()}, \code{plot.stslist()}: if \code{sortv} is a factor, the
    sequences are now sorted according to the order of the factor levels.
    \item
    \code{seqmodst()} and \code{plot.stslist.modst()}: changed name of
    attribute \code{occurences} of object returned by \code{seqmodst} to
    \code{occurrences}. Made resulting changes in \code{plot.stslsit.modst}.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqdef()}: When selecting subsets of sequence objects using \code{rownames}
    instead of row indexes, the corresponding weights were not selected.
    Fixed by setting (row)names of weight vector as the sequence
    \code{rownames}. (Alexis Gabadinho)
    }
}

}

\section{Version 1.8-2}{

\subsection{Vignettes:}{
  \itemize{
    \item
    Slightly modified JSS article vol. 40(4) added as a vignette
    on state sequence analysis.
    }
}
\subsection{Misc:}{
  \itemize{
    \item
    added \code{ex2} data sets to test and illustrate the handling of weights,
    type \code{help(ex2)} for details.
    }
}
\subsection{New functionalities:}{
  \itemize{
    \item
    New faster interface between C code and R.
    \item
    \code{seqalign()} and associated \code{print} and \code{plot} methods to see computation details
    about the alignment of two state sequences.
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqtree()}, \code{disstree()}: speed improvements.
    \item
    \code{seqtm()}: gives a warning when state names or state labels contain a comma.
    \item
    \code{seqdef()}: changed the display of alphabet, state labels and long labels when
    creating a state sequence object.
    \item
    \code{seqistatd()} and \code{seqmeant()}: added \code{prop} argument to calculate
    proportions of time spent in
    each state instead of absolute values.
    \item
    \code{seqplot()} and aliases: \code{group} now also accepts as argument a list of
    variables/vectors and produces a plot for each combination of the
    values of the variables in the list.
    \item
    \code{dissrep()}, \code{seqrep()}, \code{seqrplot()}: now accounts for weights when present.
    \item
    \code{seqtrate()}, \code{seqsubm()}: added two arguments:
      \itemize{
        \item
        \code{lag}: compute transition rates from \eqn{t} to \eqn{t+}\code{lag}, set to one by default.
        \item
        \code{with.missing}: If \code{TRUE}, compute transition rates to and from missing values.
        }
    \item
    \code{seqtreedisplay()}: now overwrites previous file if filename is not \code{NULL}.
    Tree quality measures displayed with R code.
    \item
    \code{checktriangleineq()}: internal function to check triangle inequality is now
    in C, which allows checking much bigger distance matrices.
    \item
    \code{seqIplot()}, \code{seqiplot()} and \code{seqplot()} with type \code{"I"} or \code{"i"}: the \code{sortv} argument now also accepts a sorting method,
    namely one of \code{"from.start"} or \code{"from.end"}.
    See the help page \code{?plot.stslist} for explanation.
    \item
    \code{seqeconstraint()} and other \code{seqe...} functions for event sequences: support of subsequences can now be determined by means of any of
    Joshi's 5 counting methods (see the ref manual page). The method should be
    specified with \code{seqeconstraint()}.
    \item
    \code{seqeapplaysub()}: when \code{method=NULL} is specified (now the default), the count method assigned to the event sequence object is used. With \code{method="count"} CDIST_O (number of distinct occurrences) is used as previously.
    \item
    \code{seqrep()}: attribute \code{Index} of the returned object is now a vector instead of an object of class \code{dissrep}.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqtreedisplay()}: was changing current directory when an error occurred in
    the plotting function.
    \item
    \code{dissrep()}, \code{seqrep()}: error when \code{nrep} cannot be reached (reported by M. Studer)
    \item
    \code{seqefsub()}: when using \code{strsubseq} argument, \code{countMethod} of \code{seqeconstraint} was not taken into account (Reported by Reto Bürgin).
    \item
    \code{seqeconstraint()}: added consistency checks to avoid misuses (Reported by Reto Bürgin).
    \item
    \code{print.seqelist()} and \code{as.character.seqelist()}: generated \code{segfault} when converting long event sequences to character (Reported by Pierre Molinier).
    \item
    \code{seqsubm()}: Very small rounding errors (\code{1e-16}) were sometimes leading to non symmetric substitution cost matrix (Reported by Alexandre Pollien).
    }
}
}

\section{Version 1.8-1}{

\subsection{Misc:}{
  \itemize{
    \item
    updated references in the citation file and manual pages to point to the newly published article in the \emph{Journal of Statistical Software}
    \item
    other references update in the manual pages.
    }
}

}

\section{Version 1.8}{

\subsection{New functionalities:}{
  \itemize{
    \item
    \code{seqtreedisplay()}: drawing a sequence regression tree.
    \item
    \code{seqtree()}: creating a sequence regression tree from a dissimilarity matrix.
    \item
    \code{seqrecode()}: recoding state sequences objects (i.e., merging states).
    \item
    weights are now supported by all dissimilarity analysis functions.
    \item
    weights can be assigned to event sequence objects and are
    supported by all related functions.
    \item
    \code{seqdef()}: \code{xtstep} option added to set step between displayed tick-marks and labels on the x-axis of state sequence plots.
    \item
    \code{seqplot()}: \code{xtstep} option added to state sequence plots.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqformat()}: fixed problems with \code{fillblanks} argument when converting from SPELL to STS.
    \item
    \code{seqdist()} and \code{seqdistmc()}: \code{method="HAM"} did not account for the provided substitution cost matrix; it used 1 for all substitution costs (Reported by Florian Hertel).
    \item
    \code{seqdist()}: fixed a (possible) memory leak.
    \item
    \code{seqdss()}, \code{seqdur()}: fixed bad handling of missing values in several cases:
      \itemize{
        \item
        sequences finishing with missing values,
        \item
        sequences made of only one distinct state and missing values.
        }
    \item
    \code{seqiplot()}, \code{seqIplot()}, \code{seqfplot()}: changed the automatic setting of the x-axis length, to ensure identical lengths of the x-axis when the maximal sequence length differs between groups (reported by M. Studer).
    \item
    \code{seqplot()} and aliases: fixed error with \code{xaxis} argument.
    \item
    \code{seqtransn()}: the returned normalized number of transitions for
    sequences of length 1 was \code{NaN} (value of \code{transn.norm=0/seqlength-1}). Now set to 0.
    \item
    \code{seqici()}: returns now correctly 0 instead of \code{NaN} for sequences of length 1.
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqient()}: new option \code{base} for choosing the base of the logarithm used
    to compute the entropy.
    \item
    \code{seqdist()}: enhanced check of substitution cost matrix.
      \itemize{
        \item
        The function can now be cleanly interrupted by the user.
        \item
        Timing information now uses the processor time (instead of the elapsed time).
        }
    \item
    \code{seqformat()} when converting from "SPELL":
      \itemize{
        \item
        new error message when a start time is lower than 1 and/or an end time is smaller than the start time.
        \item
        warning message when start time of episode 1 is missing (sequence creation is skipped)
        \item
        warning message when start/end time of an episode is missing (episode is skipped and filled with NA's)
        }
    \item
    \code{disstree()} and \code{dissassoc()} have been entirely redesigned; objects created with the old function are no longer supported.
    \item
    \code{seqdss()} and \code{seqdur()}: the number of columns of the returned object is now set to the maximum DSS length rather than to the length of the original state sequence object.
    }
}
\subsection{Misc:}{
  \itemize{
    \item
    \code{ex1} example data set: contains now an additional sequence \code{'s7'}
        with only missing values.
    }
}
}

\section{Version 1.6-2}{

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqdef()}: now checks whether all states encountered in the input data are present in an optional user provided alphabet (\code{state} argument)
    \item
    \code{seqefsub()}: The support of a subsequence with a total support of 1 was set to 0 (hence, this only applies if the minimum support is 1.) This is now corrected (Reported by Anna Hera).
    \item
    Corrected compilation problems under SOLARIS (Reported by Prof Brian Ripley).
    \item
    \code{seqformat()}: when converting from SPELL to STS, the \code{fillblank} argument was not used.
    }
}
}

\section{Version 1.6-1}{

\subsection{New functionalities:}{
  \itemize{
  \item
  Added a startup message when loading the TraMineR library.
  }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqtab()} and \code{seqfplot()}: \code{tlim} argument now allows to return any selected frequent sequences,
    in the same way as the \code{tlim} argument used in \code{plot.stslist()} and \code{seqiplot()} (Requested by G. Ritschard).
    For example, \code{tlim=3:6} returns the third, fourth, fifth and sixth most frequent sequences in the set. Default for \code{tlim} is now \code{1:10} instead of \code{10}.
    \item
    \code{seqsubsn()}: added detection of missing state in the sequences and computation of number of subsequences by adding missing state to the alphabet.
    \item
    \code{seqST()}: added detection of missing state in the sequences and computation of turbulence by adding missing state to the alphabet (Requested by G. Ritschard).
    \item
    \code{seqrplot} now accepts \sQuote{half} matrices (\code{dist} objects) as produced by \code{seqdist}  with the \code{full.matrix=FALSE} option (Requested by L. Lesnard).
    \item
    \code{seqiplot()} and \code{plot.stslist()}: new \code{ytlab} option allowing to display sequence labels
    on the Y-axis in sequence index plots (if set to \code{"id"}, the sequence ids are displayed).
    An additional \code{ylas} option sets the orientation of the labels (Requested by Andrew ??
    and P. Jeuniaux).
    \item
    \code{seqsubm()}:
    \itemize{
      \item
      When \code{method="TRATE"}, the substitution costs are now based on the value of \code{cval}: \eqn{SC(i,j) = cval -P(i,j) - P(j,i)} where \eqn{P(i,j)} is the transition rate from state \eqn{i} to \eqn{j}.
      \item
      added a new \code{transition} argument to use only transition from \code{"previous"} or \code{"next"} state
      instead of the default \code{"both"} when \code{time.varying = TRUE}.
      \item
      Now, by default \code{cval} equals 2, unless \code{transition} is set to \code{both} and
      \code{time.varying} is \code{TRUE} in which case \code{cval} equals 4.
      }
    \item
    \code{plot.stslist.meant()}, \code{seqmtplot()}, \code{plot.stslist.freq()}, \code{seqfplot()}, \code{plot.stslist.modst()},
    \code{seqmsplot()}: added display of weighted \eqn{n} instead of \eqn{n} in axis label if \code{weighted=TRUE}.
    \item
    \code{seqdplot()}: disabled plot of a legend for missing state if \code{with.missing=FALSE} (Requested by
      M. Studer and G. Ritschard).
    \item
    \code{seqplot()}, \code{plot.stslist()}, \code{plot.stslist.statd()}, \code{plot.stslist.modst()}, \code{plot.stslist.freq()},
    \code{plot.stslist.meant()}: when \code{weighted=TRUE}, weighted \code{n} displayed in the axis label rounded to 2 digits (Requested by M. Studer).
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{plot.stslist.modst()}, \code{seqmsplot()}: fixed bad display of missing states if \code{with.missing=TRUE}.
    \item
    \code{seqmeant()}: added color for missing state to color palette when using \code{with.missing=TRUE}.
    \item
    \code{dissrep()}, \code{seqrep()}, \code{seqrplot()}: fixed bad coverage statistics when \code{tsim} set to other than default value of 0.10 (Reported by G. Ritschard).
    \item
    \code{seqdist()}: Bad handling of missing values in DHD distance (wrong default substitution costs with missing values, set to one instead of four).
    \item
    \code{seqdist()}: Fixed an error in \code{seqdist} when using \code{refseq} with missing values (Reported by an anonymous user).
    }
}
}

\section{Version 1.6}{

\subsection{New functionalities:}{
  \itemize{
    \item
    \code{seqIplot()} and \code{seqplot(..., type="I")}: sequence index plot displaying all sequences with no space (\code{space=0}) and no border (\code{border=NA}) by default.
    \item
    \code{stlab()}: retrieving or setting the long state labels of a sequence object.
    \item
    \code{seqici()}: computes the complexity index, a composite measure of sequence complexity.
    \item
    \code{seqtransn()}: computes the number of transitions in a sequence.
    }
}
\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqrplot()} and \code{plot.stslist.rep()}: missing states are now correctly
    plotted (reported by G. Ritschard).
    \item
    \code{dissrep()} (called by \code{seqrplot()} and \code{plot.stslist.rep()}): when using coverage threshold, does no longer select one too many representatives.
    \item
    \code{seqdplot()} / \code{plot.stslist.statd()}: fixed bad coloring of missing states (reported by M. Studer).
    \item
    \code{seqstatd()} and \code{print.stslist.statd()}: fixed error when printing \code{seqstatd} output and length of the longest short state label was >2. (reported by G. Ritschard).
    \item
    \code{seqdss()}: (with default \code{with.missing=FALSE} value) DSS with identical
    successive states resulted when there were missing values between two identical states.
    For example, before: DSS of \code{"A-A-*-A-*-A"} was \code{"A-A-A"} and now \code{"A"}.
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{summary.stslist()}: added information in the output of the summary method
      for state sequence objects.
    \item
    \code{seqdist()}, \code{seqdistmc()}, \code{seqdss()}, \code{seqdur()},  \code{seqient()}, \code{seqistatd()}, \code{seqmpos()},
    \code{seqnum()}, \code{seqsubm()}:
      \code{with.miss} argument replaced by \code{with.missing} (obsolete \code{with.miss} argument
      still works for backward compatibility)
    \item
    \code{seqmeant()}: added \code{with.missing} argument to account for missing states.
    \item
    \code{seqdef()}:
    \itemize{
      \item
      new warning when one or several sequences contain only missing values.
      \item
      when no weights are provided:
      \itemize{
        \item
        \code{weights} attribute of the returned sequence object is now set to \code{NULL} instead to a vector of 1's;
        \item
        the \code{"[>] sum of weights"} message is suppressed;
        }
      \item
      new \code{Version} attribute with the number of the TraMineR version
        used for creating the sequence object;
      \item
      message \code{"[>] missing values in input file"} changed to
        \code{"[>] found missing values ('...') in sequence data"}
        and displayed only if missing values are found in the input data.
      }
    \item
    \code{seqIplot()}, seqiplot(), seqplot(..., type="I"), seqplot(..., type="i") and plot.stslist():
      new \code{weighted} argument. If set as \code{TRUE} sequence bar widths are set
        proportional to weights.
    \item
    \code{seqiplot()} and \code{plot.stslist()}: when the sequence object
    contains less than
      10 sequences and \code{tlim=NULL}, only the actual sequences are plotted
      without additional \sQuote{void sequences}.
    \item
    \code{seqmodst()}: added \code{weighted} and \code{with.missing} arguments.
    \item
    \code{seqtab()} and \code{seqstatd()}: attribute \code{nbseq} of the returned object is now
      the sum of weights (if \code{weighted=TRUE} and the sequence object has weights)
      instead of the number of sequences.
    \item
    \code{seqfplot()} / \code{plot.stslist.freq()} and \code{seqdplot()} / \code{plot.stslist.statd()}:
      when weights are used, i.e. when the \code{weighted=TRUE} argument is passed
      and the sequence object has weights,
      \itemize{
        \item
        the \sQuote{n=...} in the y axis legend now gives the sum of weights instead of
        the number of sequences;
        \item
        \sQuote{weighted} is specified in the y axis legend;
        }
    \item
    \code{seqfplot()} and \code{plot.stslist.freq()}: more precise positioning of the 0 of the y axis;
    \item
    \code{seqrplot()} and \code{plot.stslist.rep()}:
      \itemize{
        \item
        new \code{stats} option. If set to \code{FALSE}, statistics are not plotted;
        \item
        label of the yaxis changed.
        }
    \item
    \code{dissrep()}: name of the main argument changed from \code{dist.matrix} to \code{diss}  as in other \code{diss...} functions.
    }
}
}

\section{Version 1.4-1}{

\subsection{Bug fixes:}{
  \itemize{
    \item
    \code{seqecreate()}: Problems with handling simultaneous events when
    creating event sequences with data not previously sorted on
    \code{id}, \code{timestamp} and \code{event}.
    \item
    \code{seqrep()} with \code{"density"} criterion: neighborhood diameter is now
    correctly set to \code{trep*dmax} instead of \code{trep}.
    }
}
\subsection{New functionalities:}{
  \itemize{
    \item
    \code{dissrep()}: extracts a set of representative objects using a
    dissimilarity matrix. This function is used by \code{seqrep}.
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqrep()}:
      \itemize{
      \item
      much faster extraction of the representative set;
      \item
      default criterion is now \code{"density"} instead of \code{"frequency"};
      \item
      \code{trep} now sets a coverage threshold for the representative set
       rather than a size threshold for the candidate list;
      \item
      in the output object, name of the attribute containing
       statistics for the representative set changed from \code{Quality}
       to \code{Statistics} and that of the attribute containing the
       overall quality measure changed from \code{rindex} to \code{Quality}.
       }
    \item
    \code{seqplot()}: now checks that the length of the vector given as
    \code{group} argument matches the number of sequences.
    \item
    \code{plot.stslist()}: now checks that the length of the vector given as
     \code{sortv} argument matches the number of sequences.
    \item
    \code{seqformat()}: dramatic speed improvement in conversion from SPELL data.
    }
}
}

\section{Version 1.4}{

\subsection{New functionalities:}{
  \itemize{
    \item
    \code{TraMineR.checkupdates()}: check for TraMineR updates.
    \item
    \code{seqdistmc()}: computes multichannel distances.
    \item
    \code{seqmeant()}: computes mean duration in each state.
    \item
    \code{seqmodst()}: returns the sequence of modal states.
    \item
    \code{seqmsplot()}: for plotting the sequence of modal states. This
    function is a shortcut for \code{seqplot} with \code{type="ms"}, see below.
    \item
    \code{seqrep()}: extracts a set of representative sequences.
    \item
    \code{seqrplot()}: for producing representative sequence plots. This function is
    a shortcut for \code{seqplot} with \code{type="r"}, see below.
    \item
    \code{seqHtplot()}: for producing Entropy Index plots. This function is
    a shortcut for \code{seqplot} with \code{type="Ht"}, see below.
    \item
    \code{seqlogp()}: Computing logarithm of sequence probabilities.
    \item
    \code{seqdef()}: new \code{weights=} option for providing a vector of weights.
    \item
    \code{seqstatd()}, \code{seqtrate()}, \code{seqlogp()} and \code{seqtab()}: new option \code{weighted=TRUE}
    for using the weights when computing the statistics.
    \item
    \code{seqtrate()} and \code{seqsubm()}: new \code{time.varying} argument for computing
    position dependant transition rates or costs.
    \item
    \code{seqdist()}: two additional methods are now available for computing
    distances, namely \code{"HAM"} (Hamming distance) and \code{"DHD"} (Dynamic
    Hamming Distance).
    \item
    Output produced by \code{seqstatd()}, \code{seqtab()}, \code{seqmeant()}, \code{seqmodst()}, and
    \code{seqrep()} can now be plotted with their \code{plot()} dedicated methods (see
    new classes and methods below).
    }
}
\subsection{Important changes:}{
  \itemize{
    \item
    \code{seqplot()}: is now the generic function for plotting state sequence objects with
    \code{type} argument. Available types are \code{"d"} for state distribution plots,
    \code{"f"} for sequence frequency plots, \code{"Ht"} for entropy index plots,
    \code{"i"} for sequence index plots, \code{"ms"} for modal state sequence plots,
    \code{"mt"} for meant time plots, \code{"r"} for representative plots.
    This function replaces the previous generic \code{plot.stslist()} function.
    \item
    \code{plot.stslist()}: now produces only a sequence index plot (see new classes and
    methods below).
    }
}
\subsection{Changes in existing functions:}{
  \itemize{
    \item
    \code{seqfplot()}:

      \itemize{
        \item
        new \verb{yaxis} option: with \code{yaxis="cum"} (default) cumulated percentages
        are displayed, while with \code{yaxis="pct"} individual sequence percentages
        are shown.
        \item
        \code{pbarw=TRUE} is now the default for the \code{pbarw} argument.
        }

    \item
    \code{seqtab()}: the \code{format} argument that specifies the format of the sequences
    displayed as rownames is now set by default to the short SPS format,
    e.g. \code{TR/9-EM/63}.
    \item
    \code{seqiplot()}: sequence indexes are now displayed by default on the y axis.
    This can be disabled with \code{yaxis=FALSE}.
    }
}
\subsection{Fixed minor bugs in seqformat():}{
   \itemize{
      \item
      changes concern mainly the \code{from="SPELL"} and
      \code{from="SPS"} options.
   }
}
\subsection{New classes and methods:}{
  \itemize{
    \item
    new class \code{stslist.statd} for objects produced by the \code{seqstatd()} function
    and methods for printing and plotting such objects.
    \item
    new class \code{stslist.freq} for objects produced by \code{seqtab()} function and
    methods for printing and plotting such objects.
    \item
    new class \code{stslist.meant} for objects produced by \code{seqmeant()} function
    and methods for printing and plotting such objects.
    \item
    new class \code{stslist.modst} for objects produced by \code{seqmodst()} function
    and methods for printing and plotting such objects.
    }
}

}

\section{Version 1.2-1}{

\subsection{Changes in function arguments:}{
  \itemize{
    \item
    \code{seqdef()}: new \code{id} argument for setting the rownames of the sequence object.
    \item
    \code{disscenter()}: new \code{medoid.index} argument to get the indexes of all
    medoids (rather than only the first one).
    }
}
\subsection{Minor bugs fixed:}{
  \itemize{
    \item
    Plotting missing states with \code{seqiplot()} and \code{seqfplot()} functions.
    \item
    Sum of transition rates with sequences of different lengths not equal to 1.
    }
}
}

\section{Version 1.2}{

\subsection{Changes regarding plotting functions:}{
  \itemize{
    \item
    New generic function \code{plot.stlist()} with option \code{type=} for
    plotting state sequence objects of class \code{ststlist} created by
    the \code{seqdef()} function. Old functions \code{seqdplot()}, \code{seqfplot()},
    \code{seqiplot()}, \code{seqmtplot()} work as in the previous version but by
    calling \code{plot.stslist} with the appropriate \code{type=} option
    (types are \code{'d'}, \code{'f'}, \code{'i'} and \code{'mt'}). However, the order of the
    functions' arguments may have changed and this may cause problems
    if the names of the arguments were not explicitly specified in your
    scripts (which is inadvisable anyway).
    }
}
\subsection{Changes in the dissimilarity and discrepancy analysis diss module:}{
  \itemize{
    \item
    \code{dissreg()} is renamed to \code{dissmfac()} for multi-factor dissimilarity analysis.
    \item
    \code{disstree()}: great speed and memory improvements.
    }
}
\subsection{Speed and memory improvements:}{
  \itemize{
    \item
    \code{seqformat()}: conversion to TSE format is now much faster.
    \item
    \code{seqdur()}
    \item
    \code{seqST()}
    }
}
\subsection{Changes in the computation of distances between sequences:}{
  \itemize{
    \item
    \code{seqdist()} now checks if substitution costs respect the triangle inequality.
    When this is the case it ensures that the resulting dissimilarity matrix
    also respects the triangle inequality.
    \item
    New options for selecting the distance normalization method.
    \item
    New reversed LCP, i.e. longest common suffix method. (\code{method="RLCP"})
    }
}
\subsection{Various changes:}{
  \itemize{
    \item
    CITATION file added.
    \item
    \code{seqST()} caused an error when run with more than 12 states: bug fixed.
    \item
    Fixed other minor bugs.
    }
}

}

\section{Version 1.1}{

\subsection{New example data}{
  \itemize{
    \item
    \code{mvad} example data set added
    }
}
\subsection{Name changes of functions}{
  \itemize{
    \item
    \code{seqLLCP()} instead of old \code{seqLCP()}
    \item
    \code{seqLLCS()} instead of old \code{seqLCS()}
    }
}
\subsection{New module}{
New \code{diss} module for analysing a dissimilarity matrix (such as the one returned by \code{seqdist})
  \itemize{
    \item
    \code{dissassoc()}: Computes association with a factor;
    \item
    \code{dissreg()}: Regression analysis of a dissimilarity matrix;
    \item
    \code{disstree()}: Tree analysis of a dissimilarity matrix;
    \item
    \code{dissvar()}: Computes a pseudo-variance from a dissimilarity matrix.
    }
}

\subsection{Changes in graphic functions}{
 Changes in the graphics functions (seqiplot, seqfplot, seqdplot, ...):
  \itemize{
    \item
    New \code{group} option allowing to draw several plots for the levels
    of a factor within a single command;
    \item
    TraMineR now uses \code{layout} for controlling the position of the
    plots and the legend in the graphic area. This is not compatible
    with \code{par(mfrow=...)}. To use the standard \code{par(mfrow=...)} method,
    one must set the \code{use.layout=FALSE} option in the plot functions;
    \item
    \code{seqmtplot()}: new function that plots the mean time spent in each state;
    \item
    Option \code{withborder=FALSE} is now obsolete and replaced by the standard \code{border=NA} option.
    }
}

\subsection{Changes in event sequence (seqe) module:}{
  \itemize{
    \item
    New plot for exhibiting discriminant subsequences (\code{seqecmpgroup}).
    \item
    Overall syntax have been reviewed and is now much simpler.
    \item
    \code{seqefsub} now allows to search for user specified subsequences (\code{seqefsub}).
    \item
    Event subsequence lists now have specific plot and print method (\code{seqefsub}, \code{seqecmpgroup})
    \item
    \code{seqecreate} now accepts state sequences and performs automatic conversion (\code{seqecreate})
    \item
    Time constraints are now implemented separately and are stored with the results (\code{seqeconstraint})
    }
}
\subsection{Changes in the seqformat() function:}{
  \itemize{
    \item
    New options for importing SPELL formatted data.
    \item
    The STS internal and the output in STS, SPS or DSS formats are now by
    default in extended format (a matrix with one state per column) instead of
    compressed format (a character string).
    Use the \code{compressed=TRUE} option to get an output in the compressed format
    (sequences as character strings).
    \item
    \code{"SPS1"} and \code{"SPS2"} formats are now replaced by the generic \code{"SPS"} format with
    options \code{SPS.in} and \code{SPS.out} for defining the separator and surrounding
    characters used for specifying the state/duration couples.
    }
}
\subsection{Other new functions:}{
  \itemize{
    \item
    \code{seqgen()}: generates a random sequence.
    }
}
}
