"""
@Author  : Xinyi Li 李欣怡, Yuqi Liang 梁彧祺
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
        # Convert numeric codes back to state names using inverse mapping
        inverse_mapping = channels[i].inverse_state_mapping
        
        # Handle missing values: if "Missing" is in states, it has a normal mapping
        # If not, missing values (NaN) map to len(states) as the default
        # We need to ensure this code maps to the actual missing state name
        missing_code = len(channels[i].states)
        if channels[i].ismissing and missing_code not in inverse_mapping:
            # Find the missing state name (could be "Missing" or np.nan)
            missing_state = None
            for s in channels[i].states:
                if pd.isna(s) or (isinstance(s, str) and s.lower() == "missing"):
                    missing_state = s
                    break
            if missing_state is not None:
                # "Missing" is in states, so it should already be in inverse_mapping
                # But if it's not, add it
                if missing_state not in inverse_mapping.values():
                    # Find what code "Missing" maps to
                    for code, state in inverse_mapping.items():
                        if state == missing_state:
                            break
                    else:
                        # "Missing" not found in mapping, add missing_code -> missing_state
                        inverse_mapping[missing_code] = missing_state

        for j in range(maxlength):
            if j < maxlength_list[i]:
                # Convert numeric codes to state names
                # Codes are already integers from .values, but convert to int to be safe
                def code_to_state(code):
                    code_int = int(code)
                    # Use inverse mapping to get state name
                    state_name = inverse_mapping.get(code_int)
                    if state_name is None:
                        # Code not found in mapping - this shouldn't happen with valid data
                        # But handle it gracefully by returning the code as string
                        # This will help identify data issues
                        return str(code_int)
                    # Convert state name to string (handles np.nan case)
                    if pd.isna(state_name):
                        return "Missing"
                    return str(state_name)
                
                newcol = np.array([code_to_state(code) for code in seqchan[:, j]], dtype='U256')

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

            # Use the actual states from the channel (like TraMineR uses attr(channels[[i]],"alphabet"))
            # TraMineR: alphabet_list[[i]] <- attr(channels[[i]],"alphabet")
            # Convert states to strings to match what's in MD sequences
            states = [str(s) if not pd.isna(s) else "Missing" for s in channel.states]

            # Checking missing values
            if with_missing[i]:
                print("[>] Including missing value as an additional state.")
                # TraMineR adds missing value to alphabet: alphabet_list[[i]] <- c(alphabet_list[[i]],attr(channels[[i]],"nr"))
                # In SequenceData, missing values are represented internally as len(states)
                # (not len(states) + 1) because state_mapping uses 1-based indexing: states map to 1..len(states)
                # and missing values map to len(states) as the default
                # We need to add the actual missing state name (as a string) to match what's in the MD sequences
                # Find the missing state name from the states list
                missing_state = None
                for s in channel.states:
                    if pd.isna(s) or (isinstance(s, str) and s.lower() == "missing"):
                        missing_state = str(s) if not pd.isna(s) else "Missing"
                        break
                if missing_state is None:
                    missing_state = "Missing"
                # Add the missing state name to the alphabet list (as string)
                if missing_state not in states:
                    states.append(missing_state)
            else:
                if channel.ismissing:
                    raise ValueError("[!] Found missing values in channel ", i,
                                     ", set with.missing as TRUE for that channel.")

            # Check states - store as list for indexing
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
                sm_matrix = costs['sm']
                
                # TraMineR's seqcost returns a matrix WITHOUT "null" row/column
                # Python's get_substitution_cost_matrix returns DataFrame WITH "null" at index 0
                # For consistency with TraMineR, we need to remove the "null" row/column
                # when it's a DataFrame (for numpy arrays, there's no "null")
                if isinstance(sm_matrix, pd.DataFrame):
                    # Remove "null" row and column to match TraMineR structure
                    # The DataFrame has index=["null", state1, state2, ...]
                    # We want to keep only [state1, state2, ...]
                    if "null" in sm_matrix.index:
                        sm_matrix = sm_matrix.drop(index="null").drop(columns="null")
                    # Ensure DataFrame index matches alphabet_list (states) exactly
                    # This is critical for proper indexing when building CAT matrix
                    # The index should be the same as alphabet_list[i] (which is channel.states)
                    # Convert states to list of strings to ensure exact match
                    expected_index = pd.Index([str(s) for s in states])
                    current_index = pd.Index([str(s) for s in sm_matrix.index])
                    if not current_index.equals(expected_index):
                        # Reindex to match alphabet_list exactly, preserving values
                        # Use fill_value=0 for any missing entries (shouldn't happen, but safe)
                        sm_matrix = sm_matrix.reindex(index=expected_index, columns=expected_index, fill_value=0.0)
                    # Convert to numpy array for consistent indexing and to match TraMineR behavior
                    # TraMineR uses numeric indexing, so we convert DataFrame to array
                    # But keep the index mapping for .loc access
                    # Actually, keep as DataFrame for .loc access, but ensure values are correct
                
                substmat_list.append(sm_matrix)

                if "auto" == indel:
                    # costs['indel'] may include "null" at index 0 for some methods, but we only need state indels
                    # Extract state indels (skip index 0 which is "null" if present)
                    indel_val = costs['indel']
                    if isinstance(indel_val, np.ndarray) and len(indel_val) > alphsize_list[i]:
                        # Array has "null" at index 0, extract only state indels
                        state_indel = indel_val[1:]
                    elif np.isscalar(indel_val):
                        # Scalar indel, use as-is
                        state_indel = indel_val
                    else:
                        # Array with correct length (no "null" entry)
                        state_indel = indel_val
                    
                    # If it's a scalar or single-element array, repeat it for all states
                    if np.isscalar(state_indel) or (isinstance(state_indel, np.ndarray) and state_indel.size == 1):
                        indel_list.append(np.repeat(state_indel if np.isscalar(state_indel) else state_indel[0], alphsize_list[i]))
                    else:
                        # Already an array with correct length
                        indel_list.append(state_indel)
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
                        # TraMineR: ipos <- match(statelisti[chan], alphabet_list[[chan]])
                        #          cost <- cost + substmat_list[[chan]][ipos, jpos]
                        # match() returns 1-based index in R, but we use 0-based index in Python
                        state_i = statelisti[chan]  # State string from MD sequence (e.g., "1", "2")
                        state_j = statelistj[chan]  # State string from MD sequence
                        
                        # After removing "null" row/column, DataFrame structure matches TraMineR:
                        # index=[state1, state2, ...] (no "null")
                        # User-provided numpy arrays also don't have "null" row/column
                        if isinstance(substmat_list[chan], pd.DataFrame):
                            # DataFrame has structure: index=[state1, state2, ...] (no "null")
                            # Use .loc with actual state labels for safer indexing
                            # Ensure state_i and state_j are strings to match DataFrame index
                            state_i_str = str(state_i)
                            state_j_str = str(state_j)
                            if state_i_str not in substmat_list[chan].index or state_j_str not in substmat_list[chan].columns:
                                raise ValueError(f"State {state_i_str} or {state_j_str} not found in substitution matrix for channel {chan}. "
                                               f"Available indices: {list(substmat_list[chan].index)}")
                            cost += substmat_list[chan].loc[state_i_str, state_j_str]
                        else:
                            # numpy array doesn't have "null" row/column, use index directly
                            # Get 0-based index in alphabet_list
                            ipos_base = alphabet_list[chan].index(state_i)
                            jpos_base = alphabet_list[chan].index(state_j)
                            cost += substmat_list[chan][ipos_base, jpos_base]

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
                            # For time-varying matrices, there is no "null" row/column
                            # TraMineR: ipos <- match(statelisti[chan], alphabet_list[[chan]])
                            #          cost <- cost + substmat_list[[chan]][ipos, jpos, t]
                            # match() returns 1-based index in R, but we use 0-based index in Python
                            state_i = statelisti[chan]
                            state_j = statelistj[chan]
                            ipos = alphabet_list[chan].index(state_i)
                            jpos = alphabet_list[chan].index(state_j)
                            
                            # For time-varying, substmat_list[chan] is a 3D numpy array: (time, states, states)
                            # No "null" row/column, so use index directly
                            if isinstance(substmat_list[chan], np.ndarray) and substmat_list[chan].ndim == 3:
                                cost += substmat_list[chan][t, ipos, jpos]
                            else:
                                # Fallback for DataFrame (shouldn't happen for time-varying, but just in case)
                                # DataFrame has no "null" row/column after removal, use .loc with state labels
                                if isinstance(substmat_list[chan], pd.DataFrame):
                                    cost += substmat_list[chan].loc[state_i, state_j]
                                else:
                                    cost += substmat_list[chan][t, ipos, jpos]

                        newsm[t, i, j] = cost
                        newsm[t, j, i] = cost

        print("  - OK.")

        # Indel as sum
        # When newindel is None and indel is not state-dependent (simple vector), compute sum
        # TraMineR: if (is.null(newindel) & !is.list(indel_list)) newindel <- sum(indel*cweight)
        if newindel is None:
            # Check if indel is state-dependent (any element has length > 1)
            is_state_dependent = False
            if isinstance(indel, list) and len(indel) > 0:
                # Check if any indel[i] has more than one element (state-dependent)
                for ind in indel:
                    if isinstance(ind, (list, np.ndarray)) and len(ind) > 1:
                        is_state_dependent = True
                        break
            
            if is_state_dependent:
                # State-dependent indel: should have been computed above
                # If we reach here, it means we have state-dependent indels but didn't compute newindel
                # This shouldn't happen, but fallback to computing it
                newindel = np.zeros(alphabet_size)
                for i in range(alphabet_size):
                    statelisti = alphabet[i].split(ch_sep)
                    for chan in range(nchannels):
                        state = statelisti[chan]
                        ipos = alphabet_list[chan].index(state)
                        indel_val = indel[chan][ipos] if isinstance(indel[chan], (list, np.ndarray)) else indel[chan]
                        newindel[i] += indel_val * cweight[chan]
            else:
                # Simple vector: sum(indel * cweight) like TraMineR
                # Extract single values from each channel's indel
                indel_values = []
                for ind in indel:
                    if isinstance(ind, (list, np.ndarray)):
                        indel_values.append(ind[0] if len(ind) > 0 else 1.0)
                    else:
                        indel_values.append(ind)
                newindel = np.sum(np.array(indel_values) * np.array(cweight))

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
                                              states=states_space,
                                              labels=md_labels,
                                              id_col=channels[0].id_col)

            # Pass newindel as-is (can be scalar or vector depending on state-dependency)
            # TraMineR passes the full newindel vector/scalar to seqdist
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
