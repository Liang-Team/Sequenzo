\name{NEWS}
\title{NEWS file of the TraMineRextras package}

\section{Changes in version 0.7.8 [2024-08-17] }{

    \subsection{New function}{
        \itemize{
            \item
            \code{dissCompare()} comparison of two groups of data represented by a pairwise dissimilarity matrix.
        }
    }

    \subsection{Bug fixes}{
        \itemize{
            \item
            \code{seqCompare()} caused an error when only one group occurs in one of the sets. Now returns NA for such sets.
            \item
            internal \code{seq.comp()} renamed as \code{seqxcomp()}.
            \item
            missing package anchors for Rd \code{link} targets.
        }
    }
}


\section{Changes in version 0.6.7 [2024-01-08] }{

    \subsection{Changes in existing functions}{
    	\itemize{
    		\item
            All TraMineRextras plot functions: improved support of
            graphical parameters. (E.g., \code{las} and \code{cex.axis} can
            now be used to control the display of tick labels.)
    	}
    }

    \subsection{Bug fixes}{
        \itemize{
            \item
            seqimplic.Rd: lack of description for argument \code{ylim}.
            \item
            \code{seqpolyads()}: time reported when \code{core > 1} was not reliable
            \item
            \code{convert.g()}: error when \code{path=NULL}.
            \item
            \code{convert.g()} adapted to changes in ImageMagick.

        }
    }
}

\section{Changes in version 0.6.6 [2023-03-07] }{

    \subsection{New function}{
    	\itemize{
    		\item
            \code{dissindic()} case indicators to select observations that are typical, atypical,
            illustrative of or counterexample of a statistical association between trajectories
    	    and a covariate/typology.
    	}
    }
    \subsection{Changes in existing functions}{
        \itemize{
            \item
            \code{seqsplot()}, argument \code{axes} is deprecated and replaced by \code{xaxis}.
            Argument \code{yaxis} can now also be one of \code{"all"} or \code{"left"}.
            \item
            \code{seqsplot()}, \code{ylab} can be a vector of labels by group level.
            \item
            \code{seqsplot()} supports arguments of \code{legend} among the \code{...} list.
            \item
            \code{plot.stslist.surv()}, plot method for objects returned by \code{seqsurv} displays
            the ylab closer from the y-axis when \code{yaxis=FALSE}. It also recognizes the \code{cex.lab} argument.
            \item
            \code{seqsplot()}, \code{seqplot.tentrop()}, \code{seqplot.tentrop.m()}, \code{seqedplot()}, \code{plot.dynin()}, and \code{plot.stslist.surv()}: default value of \code{main} is  \code{"auto"} and \code{NULL} suppresses the titles as expected.
            \item
            \code{seplot.entrop.m()} now in namespace.
            \item
            \code{seqedplot()} argument names and behavior made conformable with those of \code{TraMineR:::seqplot}
        }
    }
}

\section{Changes in version 0.6.5 [2022-10-09] }{

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqplot.rf()} argument \code{plus.one} renamed \code{grp.meth}
    }
  }
  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqplot.rf()} \code{which.plot = "diss.to.med"} did not work.
      \item
      \code{seqplot.rf()} issue with unwanted ":" in plot title.
      \item
      \code{sorti()} possible conflict between column names of \code{seqdata} and arguments passed to \code{order}.
      \item
      issue with non UTF-8 commented line in TraMineRextras.Rd
    }
  }
  \subsection{Misc.}{
    \itemize{
      \item
      Replaced \code{permute(.)} with \code{sample(.)} in \code{seqCompare} to remove dependence on gtools that has become orphaned.
    }
  }
}


\section{Changes in version 0.6.4 [2022-06-13] }{

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqplot.rf()} gains a new argument \code{plus.one} to control which group sizes to increase when number of sequences is not a multiple of number of groups.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqindic.dyn()} checked whether \code{fstat} is a function by comparing \code{class} with a string. Now uses \code{inherits}.
      \item
      Issue with example in seqtabstocc.Rd.
      \item
      Issue with non literal item name in seqsamm.Rd.
    }
  }
}

\section{Changes in version 0.6.3 [2022-01-19] }{

  \subsection{New functions}{
    \itemize{
      \item
      \code{seqcta()} organizes data for CTA analysis (Studer et al., ALCR 2018).
      \item
      \code{seqsamm()}, \code{seqsammseq()} and \code{seqsammeha} organize and retrieve data for SAMM analysis (Studer et al., Soc. Meth. 2018).
      \item
      \code{plot.SAMM()} plot of \code{SAMM} object returned by \code{seqsamm}.
      \item
      \code{seqsha()} organizes data for SHA analysis (Rossignon et al., in Ritschard & Studer, 2018).
    }
  }

  \subsection{Misc.}{
    \itemize{
      \item
      Updated description and some Rd files.
    }

  }
}

\section{Changes in version 0.6.2 [2021-06-23]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqindic.dyn()} gains two arguments: \code{wstep} to control the spacing between successive windows and \code{endmiss.as.void} to control whether sequences end at first ending missing value or at the first void.
      \item
      \code{plot.dynin()} (plot method for objects produced by \code{seqindic.dyn}) can now plot series of summary values that end with missing values.
      \item
      \code{plot.dynin()} \code{conf} can now be logical or the confidence probability, \code{TRUE} being equivalent to 0.95.
      \item
      \code{TSE_to_STS()} now checks that \code{tmin > 0}.
      \item
      \code{seqplot.rf()} returns (invisibly) group memberships.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{plot.dynin()} issue with check of \code{fstat}.
    }
  }
}


\section{Changes in version 0.6.1 [2021-01-20]}{
  \subsection{New functions}{
    \itemize{
      \item
      \code{seqindic.dyn()} dynamic individual indicator values.
      \item
      \code{plot.dynin()} plot method for \code{dynin} objects produced
      by \code{seqindic.dyn}.
      \item
      \code{seqpolyads()} degree of similarities between polyadic members
      (contributed by Tim Liao)
      \item
      \code{polyads} example data set.
    }
  }


  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqplot.tentrop()} default color now set using
      \code{qualitative_hcl} from \code{colorspace} with the \code{'Dark 3'} palette.
    }
  }
}

\section{Changes in version 0.6.0 [2020-04-24]}{
  \subsection{New functions}{
    \itemize{
      \item
      \code{seqCompare()} BIC and Likelihood ratio test (LRT) for comparing
        two sets of sequences.
      \item
      \code{seqLRT()} LRT for comparing two sets of sequences.
      \item
      \code{seqBIC()} BIC for comparing sequence data.
    }
  }

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqgranularity()} two new methods \code{"first.valid"} and \code{"last.valid"} for sequences with missing values.
      \item
      \code{seqplot.rf()} gains a new argument \code{which.plot}.
    }
  }
  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqgranularity()} did not return a proper \code{stslist} object when \code{method="mostfreq"}.
    }
  }

}

\section{Changes in version 0.4.7 [2020-03-02]}{
  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqedplot()}: issue with color selection when only one event.
      \item
      \code{seqedplot()}: Now calls \code{plot.survfit} with by default
        \code{conf.int=FALSE} instead of \code{"none"} (advised by Terry Therneau).
    }
  }
  \subsection{Misc.}{
    \itemize{
      \item
      \code{seqprecarity(), seqprecorr(), seqprecstart()} deleted. These functions are now in TraMineR since v 2.0-13.
    }
  }
}

\section{Changes in version 0.4.6 [2019-09-06]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqedplot()}: argument \code{title} is deprecated, use \code{main} instead.
      \item
      \code{seqedplot()}: when \code{type = 'survival'}, now supports the \code{conf.int} argument of \code{plot.survfit} to control the display of confidence intervals.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqedplot()}: An issue with the joint use of the \code{xmax} and \code{xlim} arguments when calling \code{plot.survfit} raised a warning since \code{Survival} v3.0.
    }
  }
}


\section{Changes in version 0.4.5 [2019-05-06]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqprecarity()} and \code{seqprecstart()} gain an argument \code{with.missing}.
      \item
      \code{seqprecarity()}: there is now a print method for the generated \code{seqprec} object that prevents the printing of all its attributes.
      \item
      \code{seqprecorr()}: new \code{border.effect} argument. To avoid a border
        effect, when method is \code{'FREQ'}, \code{'TRATEDSS'}, or \code{'TRATE'}, the
        computed transition frequencies or probabilities \eqn{p} are adjusted
        as \eqn{p - p/}\code{border.effect} when any \eqn{p > 1 - .1/}\code{border.effect}.
      \item
      \code{seqtabstocc()} gains an argument \code{with.missing}.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqprecorr()}: \code{method='TRATE'} now works as expected.

      \item
      \code{seqprecarity()}, \code{seqprecorr()}, and \code{seqprecstart()}: errors due to bad handling of missing states.
      \item
      \code{seqedplot()} called \code{plot.survfit} with a \code{firstx} argument
        that has been suppressed (non documented in survival news);
        now uses \code{xlim} instead.

    }
  }
}

\section{Changes in version 0.4.4 [2018-12-14]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{plot.stslist.surv()}: new \code{xlab} argument.
      \item
      \code{seqprecarity()}, \code{seqprecorr()}, and \code{seqprecstart()}: Important
        code cleaning.
      \item
      \code{seqprecarity()}, \code{seqprecorr()}: New argument \code{method} that
        replaces \code{tr.type} to select the type of transition weights.
        Two additional methods have been implemented, namely \code{'RANK'}
        (for rank-based transition weights) and \code{'ONE'} for
        non-weighted transitions. Also new argument \code{stprec} to allow
        specifying rank values for \code{method='RANK'}.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqsurv()}: levels of spell$states were not ordered conformably with the alphabet of the state sequence object, which caused \code{plot.ststlist.surv} to use wrong colors for the survival curves when \code{per.state=FALSE}.
      \item
      \code{seqsplot()}: color synchronization failed when one of the states selected was unobserved in one of the groups.
      \item
      \code{seqprecorr()}: identification of non-comparable states did not work as expected.
      \item
      \code{seqprecstart()}: did not work as expected when equiv class included incomparable states.

    }
  }

}


\section{Changes in version 0.4.3 [2018-08-22]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqsplot()} now supports the \code{ncol} argument for controlling the number of columns in the color legend.
      \item
      \code{plot.stslist.surv()}, \code{seqplot.tentrop()}, , \code{seqplot.tentrop.m()}, \code{plot.seqimplic()}, \code{print.seqimplic()}: new argument \code{tick.last} that when set as \code{TRUE} allows to enforce a tick mark at the last position on the time x-axis. Has no effect when the last position is 1 + a multiple of \code{xtstep}. Default is to use the \code{tick.last} attribute of the state sequence object.
      \item
      \code{seqrep.grp()}: an error is raised when one of the groups has less than two cases.
    }
  }
  \subsection{Bug fixes}{
    \itemize{
      \item{Function \code{seqgen.missing()} the number of left and right
        elements to be turned into missings were inverted.}
    }
  }
  \subsection{Misc}{
    \itemize{
      \item{Function \code{seqeordplot()} suppressed. It is superseded by the
       function \code{seqpcplot()} of the TraMineR package.}
    }
  }
}

\section{Changes in version 0.4.2 [2018-02-01]}{
  \subsection{New functions}{
    \itemize{
      \item
      \code{seqprecarity()}: computes the sequence precarity index from
      a provided partial state order. The index is a
      corrected complexity index. See documentation page.
      \item
      \code{seqprecorr()}: determines the correction factor used by
        \code{seqprecarity()}.
      \item
      \code{seqprecstart()}: determines the starting precarity level
        of the sequence from the partial state order.
      \item
      \code{seqsurv()}: fits survival curves for states in state sequences.
        There is a plot method for objects returned by \code{seqsurv()}.
      \item
        \code{seqsplot()}: specific version of the TraMineR \code{seqplot()} function
        to plotting survival curves of states in sequences.
    }
  }

  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{seqrep.grp())} renamed argument names for consistency with TraMineR.
      \item
      \code{seqrep.grp())}: when \code{diss=NULL}, the dissimilarity matrix is now
      computed by calling \code{seqdist} with the \code{method} and possibly related parameters specified among the \code{...} arguments. An error is raised when \code{diss=NULL} and no \code{method} is provided.
    }
  }

  \subsection{Bug fixes}{
    \itemize{
      \item
      \code{seqtabstocc())}: label synchronization issues.
    }
  }

  \subsection{Misc}{
    \itemize{
      \item
      Uses function \code{TraMineR.check.depr.args()} introduced in TraMineR v 2.0-8. So, depends now on version >= 2.0-8.
    }
  }


}

\section{Changes in version 0.4.1 [2017-08-16]}{
  \subsection{Changes in existing functions}{
    \itemize{
      \item
      \code{STS_to_SPELL()}: Deleted. Converting to the \code{"SPELL"} format is now a feature
      of \code{seqformat()} in TraMineR 2.0-6+.
      \item
      \code{seqrep.grp()}: Added argument \code{with.missing} to allow computing the default distances (when \code{diss = NULL}) for sequences with missing values.
    }
  }

  \subsection{Misc}{
    \itemize{
      \item
      Replaced deprecated argument names in example of \code{seqrep.grp()}.
      \item
      Now imports from RColorBrewer, survival and cluster instead of depending on those packages.
      \item
      Suppressed unused import from combinat.
      \item
      \code{init.c} file.
    }
  }

}

\section{Changes in version 0.4.0 [2017-05-13]}{

\subsection{Note}{
  \itemize{
    \item
    TraMineRextras has been adapted to and requires now TraMineR 2.0-5.
  }
}

\subsection{Bug Fixes}{
  \itemize{
    \item
    \code{TSE_to_STS()}: The two last sequences were not converted and unexpected outcome in case of a single sequence.
    }
}

\subsection{Changes in existing functions}{
  \itemize{
    \item
    \code{seqplot.rf()}, \code{seqplot.tentrop()}, \code{seqrep.grp()}, \code{dissvar.grp()}, \code{plot.seqimplic()}: adjusted argument names (main, with.legend, legend.pos, diss, ...) to conform to the TraMineR 2 naming convention.
    }
}
\subsection{Misc}{
  \itemize{
    \item
    \code{seqrep.grp()}: Removed \code{with.missing=TRUE} in a help example as \code{biofam.seq} doesn't contain missing values and to conform to TraMineR 2.0-5+.
    \item
    \code{seqeformat()}, \code{seqentrans()}, \code{seqerulesdisc()}: Renamed '\code{seqe}' as '\code{eseq}' in internal code to conform to TraMineR 2.0-5+.
    \item
    Adjusted examples in help files (\code{sortv}, \code{seqtabstocc}, \code{seqrep.grp}, \code{seqplot.tentrop}, \code{seqplot.fr}, \code{seqimplic}, \code{seqgranularity}, \code{seqgen.missing}, \code{seqedplot}, \code{seqentrans}) to conform to TraMineR 2.0-5+.
    }
}
}

\section{Changes in version 0.2.6 [2017-03-01]}{
\subsection{Misc}{
  \itemize{
    \item{Function \code{ctplot()} suppressed. It is superseded by the
       function \code{seqpcplot()} of the TraMineR package.}
    \item{C++ routines are now registered.}
}} }

\section{Changes in version 0.2.5 [2016-11-10]}{
\subsection{New functions}{
  \itemize{
    \item{\code{seqimplic()}: Identifies position wise typical states
    of groups of sequences when compared with all the others. There
    are also plot and print methods for the outcome of the function.}
}}
}

\section{Changes in version 0.2.4 [2015-07-23]}{
\subsection{New functions}{
  \itemize{
    \item{\code{seqplot.rf()}: Implements the Relative Frequency Sequence Plots
        introduced in Fasang and Liao (2014)}
}}
\subsection{Misc}{
  \itemize{
    \item{revised help page for \code{seqemlt()} (distance of Rousset et al. 2012).}
    \item{Added required basic packages to the import statements to
        comply with R v3.3 requirements.}
}}
}
\section{Changes in version 0.2.3 [2015-01-27]}{
\subsection{New functions}{
  \itemize{
    \item{\code{seqtabstocc()}: frequencies of state co-occurrence patterns}
}}
\subsection{Changes in existing functions}{
  \itemize{
    \item{\code{seqgranularity()}: new method \code{"mostfreq"}}
    \item{\code{seqedplot()}: replaced calls of TraMineR internal functions \code{seqeage}
        and \code{TraMineR.legend} with respectively
        \code{TraMineRInternalSeqeage} and \code{TraMineRInternalLegend}
        (should be transparent to the end user).}
}}}
\section{Changes in version 0.2.2 [2013-10-30]}{
\itemize{
    \item{Fixed issue with calling Rinternals from within extern "C" {}
        in seqstart.cpp (request of B. Ripley)}
}
}
\section{Changes in version 0.2.1 [2013-03-15]}{
\subsection{General}{
  \itemize{
    \item{NEWS is now in Rd format into inst/NEWS.Rd}
}}
\subsection{Bug Fixes}{
  \itemize{
    \item{
    \code{FCE_to_TSE()}: Events were not properly chronologically sorted.}
    \item{
    \code{TSE_to_STS()}: produced inconsistent results when events were not chronologically
    sorted by cases.}
}}}
\section{Changes in version 0.2 [2012-12-06]}{
\subsection{General}{
  \itemize{
    \item{
    First release on CRAN}
}}
\subsection{Changes in existing functions}{
  \itemize{
    \item{
    \code{seqplot.tentrop()} and \code{seqplot.tentrop.m()}: new argument \code{cex.legend}
        for controlling legend scaling factor.}
    \item{
    \code{seqentrans()}: new option \code{avg.occ}; when set as \code{TRUE} the table is
        completed with average number of occurrences of the subsequences
        in each sequence. The latter differs from the support for
        counting methods other than COBJ. Requires TraMineR 1.8-5 or higher.}
    \item{
    \code{dissvar.grp()} and \code{seqrep.grp()} now also accept distance objects as \code{mdis} argument.}
    \item{
    \code{group.p} has now a \code{weights} argument.}
}}
\subsection{Bug Fixes}{
  \itemize{
    \item{
    \code{seqedist()}: fixed a mix between norm and interval arguments.}
    \item{
    fixed build warning generated by calls of foreign functions.}
}}}
