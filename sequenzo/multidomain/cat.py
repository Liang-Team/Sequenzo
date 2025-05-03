"""
@Author  : 李欣怡
@File    : cat.py
@Time    : 2025/4/8 09:06
@Desc    : Build multidomain (MD) sequences of combined individual domain states (expanded alphabet),
           derive multidomain indel and substitution costs from domain costs by means of an additive trick (CAT),
           and compute OM pairwise distances using CAT costs.
"""
import numpy as np
import pandas as pd
from typing import List, Union, Optional
import contextlib
import io

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils import seqlength
from sequenzo.dissimilarity_measures import get_distance_matrix, get_substitution_cost_matrix


def compute_cat_distance_matrix(channels: List[SequenceData],
                                method: Optional[str] = None,
                                norm: str = "none",
                                indel: Union[float, np.ndarray, List[Union[float, List[float]]]] = "auto",
                                sm: Optional[Union[List[str], List[np.ndarray]]] = None,
                                with_missing: Optional[Union[bool, List[bool]]] = None,
                                full_matrix: bool = True,
                                link: str = "sum",
                                cval: float = 2,
                                miss_cost: float = 2,
                                cweight: Optional[List[float]] = None,
                                what: str = "MDseq",
                                ch_sep: str = "+"):
    """
            mulitdomain sequences analysis, you can get:
            - multi-domain sequences ('MDseq')
            - multi-domain substitution and indel costs ('cost')
            - multi-domain distance_matrix ('diss')

            :param channels: A list of domain state sequence stslist objects defined with the define_sequences_data function
            :param method: Dissimilarity measure between sequences.
            :param norm: The normalization method to use. Ignored if what is not "diss".
            :param indel: An insertion/deletion cost or a vector of state dependent indel costs for each domain.
            :param sm: A list with a substitution-cost matrix for each domain,
                       or a list of method names for generating the domain substitution costs
            :param with_missing: Whether consider missing values
            :param full_matrix: the full distance matrix between MD sequences is returned.
            :param link: Method to compute the "link" between domains.
            :param cval: Domain substitution cost for "CONSTANT" matrix, for seqcost
            :param miss_cost: Cost to substitute missing values at domain level, for seqcost
            :param cweight: A vector of domain weights.
            :param what: What output should be returned?
            :param ch_sep: Separator used for building state names of the expanded alphabet.
    """

    # ==================
    # Checking Arguments
    # ==================
    if what == "sm":
        print("[!] what='sm' deprecated! Use what='cost' instead.")
        what = "cost"
    elif what == "seqmc":
        print("[!] what='seqmc' deprecated! Use what='MDseq' instead.")
        what = "MDseq"

    valid_whats = ["MDseq", "cost", "diss"]
    if what not in valid_whats:
        raise ValueError(f"[!] 'what' should be one of {valid_whats}.")

    if what == "diss" and not method:
        raise ValueError("[!] A valid 'method' must be provided when what = 'diss'.")
    if what == "cost" and sm is None:
        raise ValueError("[!] 'sm' cannot be NULL when what = 'cost'.")

    nchannels = len(channels)
    if nchannels < 2:
        raise ValueError("[!] Please specify at least two domains.")

    # Check cweight
    if cweight is None:
        cweight = np.repeat(1.0, nchannels)

    # If time varying sm are provided, all sm must be 3-dimensional
    timeVarying = False
    if isinstance(sm, list) and isinstance(sm[0], np.ndarray):
        ndims = [arr.ndim for arr in sm]
        if any(d == 3 for d in ndims) and not all(d == 3 for d in ndims):
            raise ValueError("[!] One sm is 3-dimensional and some are not.")

        if ndims[0] == 3:
            timeVarying = True

    # Check indel
    # Convert all elements in indel(list) to list
    if isinstance(indel, (float, int)):
        indel = [indel] * nchannels
        indel = [[x] for x in indel]

    if isinstance(indel, np.ndarray):
        indel = [[x] for x in indel.tolist()]

    if isinstance(indel, list) and isinstance(indel[0], (float, int)):
        indel = [[x] for x in indel]

    if len(indel) > 1 and any(indel == "auto" for indel in indel):
        raise ValueError("[!] 'auto' not allowed in vector or list indel.")

    if isinstance(indel, list) and len(indel) == 1:
        raise ValueError("[!] When a list or vector, indel must be of length equal to number of domains.")

    if isinstance(indel, list) and len(indel) != nchannels:
        raise ValueError("[!] When a list or vector, indel must be of length equal to number of domains.")

    # Check missing
    has_miss = np.repeat(False, nchannels)

    for i in range(nchannels):
        channel = channels[i]
        alphabet = channel.states

        # Check separator
        if any(ch_sep in str(s) for s in alphabet):
            raise ValueError(f"[!] 'ch.sep' symbol ({ch_sep}) occurs in alphabet of at least one channel.")

        has_miss[i] = channel.ismissing
        if with_missing is not None and has_miss[i] != with_missing[i]:
            with_missing[i] = has_miss[i]
            print(f"[!] Bad with.missing value for domain {i + 1}. I set it as {has_miss[i]}.")

    if with_missing is None:
        with_missing = has_miss

    if isinstance(with_missing, bool) or len(with_missing) == 1:
        with_missing = np.repeat(with_missing, nchannels)

    if len(with_missing) > 1 and len(with_missing) != nchannels:
        raise ValueError("[!] When a vector, with.missing must be of length equal to number of domains.")

    # Check number of sequences for each channel
    first_len = channels[0].seqdata.shape[0]
    if not all(channel.seqdata.shape[0] == first_len for channel in channels):
        raise ValueError("[!] sequence objects have different numbers of rows.")

    numseq = first_len

    print(f"[>] {nchannels} domains with {numseq} sequences.")
    # Actually LCP and RLCP are not included

    # Check what : method, sm
    if what == "diss":
        metlist = ["OM", "LCS", "DHD", "HAM"]

        if method not in metlist:
            raise ValueError(f"[!] 'method' should be one of {metlist}.")
        if not isinstance(sm, list):
            raise ValueError(f"[!] 'sm' should be a list.")

        if method == "LCS":
            method = "OM"
            sm = "CONSTANT"
            indel = list(np.repeat(indel, nchannels))
            cval = 2
            miss_cost = 2

        timeVarying = method == "DHD"

        if sm is None:
            costmethod = "CONSTANT"
            if method == "DHD":
                costmethod = "TRATE"
            sm = list(np.repeat(costmethod, nchannels))

    if len(sm) == 1 and sm[0] in ["CONSTANT", "TRATE", "INDELS", "INDELSLOG"]:
        sm = list(np.repeat(sm, nchannels))

    # Checking correct numbers of info per channel
    if what != "MDseq":
        if len(sm) != nchannels or len(cweight) != nchannels:
            raise ValueError("[!] You must supply one weight, one substitution matrix, and one indel per domain.")

    # Checking that all channels have the same length
    slength1 = seqlength(channels[1])
    for i in range(1, nchannels):
        if not np.array_equal(slength1, seqlength(channels[i])):
            print("[!] Cases with sequences of different length across domains.")
            break

    substmat_list = []  # subsitution matrix
    indel_list = []  # indels per channel
    alphabet_list = []  # alphabet for each channel
    alphsize_list = []  # alphabet size per channel
    maxlength_list = np.zeros(nchannels)  # seqlenth of each channels

    # Storing number of columns and cnames
    for i in range(nchannels):
        maxlength_list[i] = channels[i].seqdata.shape[1]
    max_index = np.argmax(maxlength_list)
    md_cnames = channels[max_index].seqdata.columns

    print("[>] Building MD sequences of combined states.")

    # ================================
    # Building the New Sequence Object
    # ================================
    maxlength = int(np.max(maxlength_list))
    newseqdata = np.full((numseq, maxlength), "", dtype='U256')

    for i in range(nchannels):
        seqchan = channels[i].values.copy()

        for j in range(maxlength):
            if j < maxlength_list[i]:
                newcol = seqchan[:, j].astype(str)

                # TraMineR default missing value is legal, and we already do this.
                # newseqdataNA[,j] <- newseqdataNA[,j] & newCol == void

                # SequenceData has no attributes void, so we default fill with missing value (np.nan)
                # if (fill.with.miss == TRUE & has.miss[i] & any(newCol == void)) {
                #     newCol[newCol == void] < - nr
                # }

            else:
                newcol = np.repeat("", numseq)

            if i > 0:
                newseqdata[:, j] = np.char.add(np.char.add(newseqdata[:, j], ch_sep), newcol)
            else:
                newseqdata[:, j] = newcol

    states_space = list(np.unique(newseqdata))

    print("  - OK.")

    if what == "MDseq":
        return newseqdata
    else:
        # ==================================================
        # Building Substitution Matrix and Indel Per Channel
        # ==================================================
        for i in range(nchannels):
            channel = channels[i]

            if not isinstance(channel, SequenceData):
                raise ValueError("[!] Channel ", i,
                                 " is not a state sequence object, use 'seqdef' function to create one.")

            # Since states is prepared for the upcoming MD states
            # And MD uses numeric representations, we use numbers here instead of the original string states.
            states = np.arange(1, len(channel.states) + 1).astype(str).tolist()

            # Checking missing values
            if with_missing[i]:
                print("[>] Including missing value as an additional state.")
            else:
                if channel.ismissing:
                    raise ValueError("[!] Found missing values in channel ", i,
                                     ", set with.missing as TRUE for that channel.")

            # Check states
            alphabet_list.append(states)
            alphsize_list.append(len(states))

            # Pre-progress indel
            if indel != "auto" and len(indel[i]) == 1:
                indel[i] = np.repeat(indel[i], alphsize_list[i])

            # Substitution matrix generation method is given
            if isinstance(sm[i], str):
                print(f"[>] Computing substitution cost matrix for domain {i}.")

                with contextlib.redirect_stdout(io.StringIO()):
                    costs = get_substitution_cost_matrix(channel, sm[i],
                                                         with_missing=has_miss[i],
                                                         time_varying=timeVarying, cval=cval,
                                                         miss_cost=miss_cost)
                substmat_list.append(costs['sm'])

                if "auto" == indel:
                    costs['indel'] = np.repeat(costs['indel'], alphsize_list[i])
                    indel_list.append(costs['indel'])
                else:
                    indel_list.append(indel[i])

            else:  # Provided sm
                substmat_list.append(sm[i])

                if "auto" == indel:
                    indel_list.append(np.repeat(np.max(sm[i]) / 2, alphsize_list[i]))
                else:
                    indel_list.append(indel[i])

            # Mutliply by channel weight
            substmat_list[i] = cweight[i] * substmat_list[i]

        if "auto" == indel:
            indel = indel_list

        # =============================================
        # Building the New CAT Substitution Cost Matrix
        # =============================================
        print("[>] Computing MD substitution and indel costs with additive trick.")

        # Build new subsitution matrix and new alphabet
        alphabet = states_space
        alphabet_size = len(alphabet)
        newindel = None

        # Recomputing the substitution matrix
        if not timeVarying:
            newsm = np.zeros((alphabet_size, alphabet_size))
            newindel = np.zeros(alphabet_size)

            # To reduce redundancy, we simply merged the code for retrieving sm and indel
            statelisti = alphabet[alphabet_size - 1].split(ch_sep)
            for i in range(nchannels):
                state = statelisti[i]
                ipos = alphabet_list[i].index(state)

                newindel[alphabet_size - 1] += indel[i][ipos] * cweight[i]

            for i in range(alphabet_size - 1):
                statelisti = alphabet[i].split(ch_sep)

                for chan in range(nchannels):
                    state = statelisti[chan]
                    ipos = alphabet_list[chan].index(state)

                    newindel[i] += indel[chan][ipos] * cweight[chan]

                for j in range(i + 1, alphabet_size):
                    cost = 0
                    statelistj = alphabet[j].split(ch_sep)

                    for chan in range(nchannels):
                        ipos = alphabet_list[chan].index(statelisti[chan]) + 1
                        jpos = alphabet_list[chan].index(statelistj[chan]) + 1
                        cost += substmat_list[chan].iloc[ipos, jpos]

                    newsm[i, j] = cost
                    newsm[j, i] = cost

        else:
            # Recomputing time varying substitution
            newsm = np.zeros((maxlength, alphabet_size, alphabet_size))

            for t in range(maxlength):
                for i in range(alphabet_size - 1):
                    statelisti = alphabet[i].split(ch_sep)

                    for j in range(i + 1, alphabet_size):
                        cost = 0
                        statelistj = alphabet[j].split(ch_sep)

                        for chan in range(nchannels):
                            ipos = alphabet_list[chan].index(statelisti[chan])
                            jpos = alphabet_list[chan].index(statelistj[chan])

                            cost += substmat_list[chan][t, ipos, jpos]

                        newsm[t, i, j] = cost
                        newsm[t, j, i] = cost

        print("  - OK.")

        # Indel as sum
        if newindel is None:
            newindel = np.sum(cweight * cweight[:, np.newaxis], axis=0)

        # If we want the mean of cost
        if link == "mean":
            newindel = newindel / np.sum(cweight)
            newsm = newsm / np.sum(cweight)

        if what == "cost":
            return {
                "sm": newsm,
                "indel": newindel,
                "alphabet": alphabet,
                "cweight": cweight
            }

        if what == "diss":
            if np.any(np.isnan(newsm)) or np.any(np.isnan(newindel)):
                raise ValueError("NA values found in substitution or indel costs. Cannot compute MD distances.")

            print("[>] Computing MD distances using additive trick.")

            # This step will hide the state concatenation,
            # And the returned result will convert the MD strings into numbers.
            # for example : '1+2+3' --> 1, '1+4+6' --> 2
            newseqdata_df = pd.DataFrame(newseqdata, columns=md_cnames)
            newseqdata_df.insert(0, channels[0].id_col, channels[0].ids)

            # Reconstruct multi-domain labels for composite states
            domain_labels = [channel.labels for channel in
                             channels]  # e.g., [["At home", "Left home"], ["No child", "Child"]]

            md_labels = []
            for md_state in states_space:
                parts = md_state.split(ch_sep)  # e.g., ["0", "1"]
                if len(parts) != len(domain_labels):
                    md_labels.append(md_state)  # fallback if structure doesn't match
                else:
                    label_parts = []
                    for val, dom_lab in zip(parts, domain_labels):
                        try:
                            label_parts.append(dom_lab[int(val)])
                        except (ValueError, IndexError):
                            label_parts.append(str(val))  # fallback if unexpected value
                    md_labels.append(" + ".join(label_parts))

            with contextlib.redirect_stdout(io.StringIO()):
                newseqdata_seq = SequenceData(newseqdata_df,
                                              time=md_cnames,
                                              time_type=channels[0].time_type,
                                              states=states_space,
                                              labels=md_labels,
                                              id_col=channels[0].id_col)

            newindel = np.max(newindel)
            with contextlib.redirect_stdout(io.StringIO()):
                diss_matrix = get_distance_matrix(newseqdata_seq,
                                                  method=method,
                                                  norm=norm,
                                                  indel=newindel,
                                                  sm=newsm,
                                                  with_missing=False,
                                                  full_matrix=full_matrix)
            print("  - OK.")

            diss_matrix = pd.DataFrame(diss_matrix, index=channels[0].ids, columns=channels[0].ids)
            return diss_matrix


if __name__ == '__main__':
    # from sequenzo import *
    #
    # # df = pd.read_csv("D:/college/research/QiQi/sequenzo/files/sampled_data_sets/broad_data/sampled_30000_data.csv")
    # # df = pd.read_csv("D:/college/research/QiQi/sequenzo/files/orignal data/detailed_sequence_10_work_years_df.csv")
    # # df = pd.read_csv("D:/college/research/QiQi/sequenzo/seqdef/sampled_data_1000.csv")
    # df = pd.read_csv("D:/college/research/QiQi/sequenzo/files/sampled_data_sets/detailed_data/sampled_1000_data.csv")
    #
    # # df = pd.read_csv("D:/country_co2_emissions_missing.csv")
    #
    # time = list(df.columns)[4:]
    # # time = list(df.columns)[1:]
    #
    # # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # # states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    #
    # sequence_data = SequenceData(df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']],
    #                              time_type="age", time=time, id_col="worker_id", states=states)
    # # sequence_data = SequenceData(df, time_type="age", time=time, id_col="country", states=states)
    #
    # sequence_data = [sequence_data, sequence_data]
    #
    # MD = seqMD(sequence_data, method="OM", sm=["TRATE"], indel="auto", what="diss", link="mean")
    # print(MD)
    #
    # print("================")

    from sequenzo import *

    left_df = load_dataset('biofam_left_domain')
    children_df = load_dataset('biofam_child_domain')
    married_df = load_dataset('biofam_married_domain')

    time_cols = [col for col in children_df.columns if col.startswith("age_")]

    seq_left = SequenceData(data=left_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["At home", "Left home"])
    seq_child = SequenceData(data=children_df, time_type="age", time=time_cols, states=[0, 1],
                             labels=["No child", "Child"])
    seq_marr = SequenceData(data=married_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["Not married", "Married"])

    sequence_data = [seq_left, seq_child, seq_marr]

    cat_distance_matrix = compute_cat_distance_matrix(sequence_data, method="OM", sm=["TRATE"], indel=[2, 1, 1], what="diss", link="sum")

    print(cat_distance_matrix)

    print("================")
