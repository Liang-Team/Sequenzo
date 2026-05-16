"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_analysis_multi_state_model.py
@Time    : 30/09/2025 20:27
@Desc    : Sequence Analysis Multi-state Model (SAMM) for event history analysis
           
           This module provides tools for analyzing sequences through a multi-state perspective,
           creating person-period datasets that can be used for event history analysis.
           
           Port of the SAMM workflow from the R package **TraMineRextras** (Studer et al.;
           not TraMineR core). Input sequences are expected as Sequenzo ``SequenceData``
           (analogous to TraMineR ``seqdef()`` objects).

R package roles
---------------
- **TraMineRextras**: ``seqsamm()``, ``seqsammseq()``, ``seqsammeha()``, S3 ``plot.SAMM()``
- **TraMineR** (dependency only): ``seqdef()`` for input; ``seqplot()`` inside ``plot.SAMM()``

Python ↔ R function map
-----------------------
+-------------------------------+------------------------------------------+
| Python (Sequenzo)             | R (TraMineRextras)                       |
+-------------------------------+------------------------------------------+
| ``sequence_analysis_multi_state_model`` | ``seqsamm()``                    |
| ``seqsamm`` (alias)             | ``seqsamm()``                            |
| ``SAMM``                        | object of class ``"SAMM"``               |
| ``plot_samm()``                 | ``plot.SAMM()`` (calls TraMineR ``seqplot()``) |
| ``seqsammseq()``                | ``seqsammseq()``                         |
| ``set_typology()``              | no full equivalent (R assigns typology  |
|                               | inside ``seqsammeha()``; see note below) |
| ``seqsammeha()``                | ``seqsammeha()``                         |
+-------------------------------+------------------------------------------+

References: TraMineRextras ``?seqsamm``;
  source: https://rdrr.io/cran/TraMineRextras/src/R/seqsamm.R
  manual: https://cran.r-project.org/web/packages/TraMineRextras/refman/TraMineRextras.html

Column naming: primary column is ``spell.time`` (R); ``spell_time`` is kept as an alias.
Subsequence columns ``s.1``, ``s.2``, ... store numeric state codes (TraMineR-style), not labels.

Void vs missing (``seqsamm`` row filtering)
-------------------------------------------
TraMineR ``seqdef()`` stores a **void** symbol (default ``"%"``) in ``attr(seqdata, "void")``,
separate from **missing** (``nr``). TraMineRextras ``seqsamm()`` keeps a person-period row only
if the subsequence has **no void** in any of its positions (``maxmiss = 0`` when
``minlength == sublength``).

Sequenzo ``SequenceData`` accepts ``void=`` (default ``"%"``, like ``seqdef()``) and sets
``seqdata.void_code`` when the void symbol is listed in ``states``. ``seqsamm`` drops a
person-period row if any position in the subsequence equals ``void_code``. Pass
``void=None`` at construction to disable void metadata and void-based row dropping.

The automatic **Missing** state that ``SequenceData`` may append is **not** the same as
void. Rows with missing codes in the subsequence are **not** dropped by ``seqsamm`` unless
you deliberately set ``void`` to the missing symbol (not recommended).

For numerical parity with R, run the **same** ``SequenceData`` / ``seqdef`` object through
``seqsamm`` / ``sequence_analysis_multi_state_model`` and compare row count, ``time`` range,
and ``spell.time`` (see ``samm_examples.md``).

IMPORTANT DIFFERENCES FROM R'S ``plot.SAMM`` IMPLEMENTATION:

Plotting approach (Python ``plot_samm`` vs R ``plot.SAMM``):

R ``plot.SAMM()`` (TraMineRextras, defined in ``seqsamm.R``):
  - Uses TraMineR ``seqplot()`` with grouping by the state transitioned out of
  - Original R code: plot.SAMM <- function(x, type="d", ...){
                       seqdata <- attr(x, "stslist")[x$transition,]
                       group <- x[x$transition, attr(x, "sname")[1]]
                       levels(group) <- paste("Transition out of", levels(group))
                       seqplot(seqdata, group=group, type=type, ...)
                     }
  - Creates grouped sequence plots where sequences are grouped by starting state
  - Relies on TraMineR's built-in plotting system

Our Python ``plot_samm()`` function:
  - Uses matplotlib's imshow() with sequence index plot approach
  - Creates separate subplots for each starting state (one subplot per transition state)
  - Each subplot shows all subsequences that start with a specific state as colored horizontal bars
  - Displays actual sequence patterns using a color-coded matrix visualization
  - Automatically handles varying numbers of sequences per state with dynamic subplot heights

Why We Made This Choice:
  1. Better Visual Separation: Each starting state gets its own dedicated subplot,
     making it easier to compare patterns across different states
  2. Scalability: Works well with large numbers of sequences and states
  3. Clarity: Direct visualization of subsequence patterns without grouping artifacts
  4. Python Ecosystem: Leverages matplotlib's powerful visualization capabilities
  5. Detail Preservation: Shows individual sequence patterns rather than aggregate summaries

Both approaches show transition patterns effectively, but our Python implementation 
provides more detailed, subplot-based visualizations that are particularly suitable
for exploratory data analysis and detailed pattern inspection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt

# Import the SequenceData class from the parent package
from sequenzo.define_sequence_data import SequenceData

# R column name (TraMineRextras seqsamm); spell_time is a backward-compatible alias.
SPELL_TIME_COL = "spell.time"
SPELL_TIME_ALIAS = "spell_time"


def _state_to_code_map(seqdata: SequenceData) -> Dict:
    """Map state symbols to integer codes (inverse of ``inverse_state_mapping``)."""
    inv = seqdata.inverse_state_mapping
    return {state: int(code) for code, state in inv.items()}


def _void_codes_for_samm(seqdata: SequenceData) -> set:
    """
    Integer codes treated as void in subsequences (TraMineR ``attr(seqdata, "void")``).

    Uses ``seqdata.void_code`` when set (``void`` symbol in ``states``); otherwise no
    void-based row dropping. See ``SequenceData(void=...)`` and ``samm_examples.md``.

    With ``minlength == sublength``, R sets ``maxmiss = 0``: any void in the
    subsequence drops that person-period row.
    """
    code = getattr(seqdata, "void_code", None)
    if code is not None:
        return {int(code)}
    return set()


def _subseq_rows_valid(subseq: np.ndarray, void_codes: set) -> np.ndarray:
    """True for rows with no void code in any subsequence position (R ``cond``)."""
    if not void_codes:
        return np.ones(subseq.shape[0], dtype=bool)
    invalid = np.isin(subseq, list(void_codes)).any(axis=1)
    return ~invalid


def _covar_rows_for_ids(covar: pd.DataFrame, id_values: np.ndarray) -> pd.DataFrame:
    """
    Align ``covar`` rows to sequence ids (R: ``covar[ret$id, ]``).

    Requires index labels to match ``id_values`` in value and dtype (e.g. both int or both str).
    """
    ids = np.asarray(id_values)
    id_index = pd.Index(ids)
    covar = covar.copy()

    if covar.index.equals(id_index):
        return covar.loc[id_index]

    covar_str = pd.Index(covar.index.astype(str))
    id_str = pd.Index(ids.astype(str))
    if covar_str.equals(id_str):
        raise ValueError(
            "covar index values match sequence ids as strings but dtypes differ "
            f"(covar index {covar.index.dtype!r} vs sequence ids {id_index.dtype!r}). "
            "Cast covar.index to the same type as sequence ids, as required for R "
            "rownames(covar) == id (e.g. both int or both str)."
        )

    try:
        cast_index = covar.index.astype(id_index.dtype, copy=False)
        covar_cast = covar.copy()
        covar_cast.index = cast_index
        if cast_index.equals(id_index):
            return covar_cast.loc[id_index]
    except (TypeError, ValueError):
        pass

    if len(covar.index) != len(set(covar.index)):
        raise ValueError(
            "covar index contains duplicate row labels; each sequence id must appear once."
        )

    missing_ids = set(ids) - set(covar.index)
    if missing_ids:
        sample = list(missing_ids)[:5]
        extra = list(set(covar.index) - set(ids))[:5]
        extra_msg = f" Extra covar index labels (examples): {extra}." if extra else ""
        raise ValueError(
            f"covar index is missing sequence ids (examples: {sample}). "
            f"Row index must match sequence ids (R: covar[ret$id, ]). "
            f"covar index dtype={covar.index.dtype!r}, sequence id dtype={id_index.dtype!r}."
            f"{extra_msg}"
        )

    return covar.loc[id_index]


def _remap_codes_for_imshow(matrix: np.ndarray, code_to_idx: Dict[int, int]) -> np.ndarray:
    """Map state codes to contiguous 0..K-1 indices for matplotlib colormaps."""
    out = np.full(matrix.shape, np.nan, dtype=float)
    for code, idx in code_to_idx.items():
        out[matrix == code] = float(idx)
    return out


def _resolve_spell_value(
    spell: Union[str, int, float],
    state_to_code: Dict,
    labels: List[str],
    alphabet: List,
) -> Union[int, float, str]:
    """
    Resolve ``spell`` to the value stored in ``s.1`` (numeric state code).

    Accepts TraMineR-style state codes, state symbols (``alphabet``), or labels.
    """
    if isinstance(spell, (int, np.integer)) or (
        isinstance(spell, float) and spell == int(spell) and not np.isnan(spell)
    ):
        return int(spell)
    if isinstance(spell, str) and spell.isdigit():
        return int(spell)
    if spell in state_to_code:
        return state_to_code[spell]
    label_to_state = dict(zip(labels, alphabet))
    if spell in label_to_state:
        state = label_to_state[spell]
        return state_to_code[state]
    raise ValueError(
        f"spell {spell!r} not found among state codes, states {alphabet}, or labels {labels}"
    )


def _typology_type_levels(
    typology: Union[pd.Series, np.ndarray, list, None],
    labels_array: Optional[np.ndarray],
) -> List:
    """Match R ``seqsammeha`` dummy column names: factor levels or unique values."""
    if typology is None:
        if labels_array is None:
            return []
        return list(pd.unique(labels_array))

    if isinstance(typology, pd.Series) and isinstance(typology.dtype, pd.CategoricalDtype):
        return list(typology.cat.categories)

    if isinstance(typology, pd.Categorical):
        return list(typology.categories)

    return list(pd.unique(np.asarray(typology, dtype=object)))


def _build_typology_labels(
    samm: "SAMM",
    spell_code: Union[int, float],
    *,
    typology: Union[pd.Series, np.ndarray, list, None] = None,
    clusters: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cluster_to_name: Optional[Dict] = None,
    mapping: Optional[Union[Dict, pd.Series]] = None,
    by: Optional[str] = None,
) -> np.ndarray:
    """Row-aligned typology labels for transition rows (does not modify ``samm``)."""
    condition = (samm.data["s.1"] == spell_code) & (samm.data["transition"].astype(bool))
    n_transitions = int(condition.sum())

    if typology is not None:
        labels_array = (
            typology.values if isinstance(typology, pd.Series) else np.asarray(typology, dtype=object)
        )
    elif clusters is not None:
        clusters_array = clusters.values if isinstance(clusters, pd.Series) else np.asarray(clusters)
        if len(clusters_array) != n_transitions:
            raise ValueError(
                f"clusters length {len(clusters_array)} must match n_transitions={n_transitions}"
            )
        if cluster_to_name is not None:
            labels_array = np.asarray([cluster_to_name[c] for c in clusters_array], dtype=object)
        else:
            labels_array = clusters_array.astype(object)
    elif mapping is not None:
        labels_array = _expand_typology_for_transitions(
            samm=samm,
            spell=spell_code,
            mapping=mapping,
            by=by,
            cluster_to_name=cluster_to_name,
        )
    else:
        raise ValueError("Provide typology, clusters, or mapping.")

    if len(labels_array) != n_transitions:
        raise ValueError(
            f"typology length {len(labels_array)} must match n_transitions={n_transitions}"
        )
    return labels_array


class SAMM:
    """
    Sequence Analysis Multi-state Model (SAMM) object.
    
    Corresponds to the R S3 class ``"SAMM"`` returned by TraMineRextras ``seqsamm()``.
    
    This class stores a person-period dataset generated from sequence data,
    where each row represents one time point for one person, along with
    information about subsequences, transitions, and spell characteristics.
    
    Attributes:
        data (pd.DataFrame): The person-period dataset
        alphabet (list): The state space (unique states in the sequences)
        labels (list): Labels for the states
        color_map (dict): Color mapping for visualization
        sname (list): Column names for subsequence variables (e.g., ['s.1', 's.2', 's.3'])
        sublength (int): Length of the subsequences being tracked
        state_to_code (dict): Map state symbols to integer codes (for ``spell`` arguments)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        alphabet: list,
        labels: list,
        color_map: dict,
        sname: list,
        sublength: int,
        state_to_code: Optional[Dict] = None,
    ):
        """
        Initialize a SAMM object.
        
        Args:
            data: Person-period dataset
            alphabet: List of unique states
            labels: Labels for states
            color_map: Dictionary mapping states to colors
            sname: List of subsequence column names
            sublength: Length of subsequences
            state_to_code: State symbol to numeric code (from ``SequenceData``)
        """
        self.data = data
        self.alphabet = alphabet
        self.labels = labels
        self.color_map = color_map
        self.sname = sname
        self.sublength = sublength
        self.state_to_code = state_to_code or {}
        
        # Initialize typology column (will be set later using set_typology)
        if 'typology' not in self.data.columns:
            self.data['typology'] = 'None'
    
    def __repr__(self):
        """String representation of SAMM object."""
        return f"SAMM(n_rows={len(self.data)}, sublength={self.sublength})"
    
    def __len__(self):
        """Return number of rows in the person-period dataset."""
        return len(self.data)


def sequence_analysis_multi_state_model(seqdata: SequenceData, sublength: int, covar: Optional[pd.DataFrame] = None) -> SAMM:
    """
    Generate a person-period dataset from sequence data for multi-state analysis.
    
    R equivalent: **TraMineRextras** ``seqsamm(seqdata, sublength, covar = NULL)``.
    ``seqdata`` should be a TraMineR ``seqdef`` object in R; here, pass a Sequenzo
    ``SequenceData`` object (built the same way as for other Sequenzo sequence tools).
    
    This function transforms sequence data into a "person-period" format where each row 
    represents one time point for one individual. At each time position, it also extracts
    the subsequence for the next 'sublength' time units.
    
    **What is person-period data?**
    Instead of having one row per person with all their time points as columns,
    person-period data has one row for each person-time combination. For example,
    if we track 3 people over 5 time periods, we get 15 rows (3 x 5).
    
    **What are subsequences?**
    At each time point, we look ahead and record what happens in the next few time periods.
    For example, if sublength=3 and we're at time 2, we record states at time 2, 3, and 4.
    
    Args:
        seqdata (SequenceData): A SequenceData object containing your sequence data.
                                This should be created using the SequenceData class.
        sublength (int): The length of the subsequence to extract at each time point.
                        For example, if sublength=3, we look 3 steps ahead from each position.
        covar (pd.DataFrame, optional): Time-invariant covariates (variables that don't change over time).
                                       For example: gender, education level, birth year, etc.
                                       The row index should match the sequence IDs.
    
    Returns:
        SAMM: A SAMM object containing the person-period dataset with the following variables:
            - id: Identifier for each sequence/person
            - time: Time elapsed since the beginning of the sequence (starts at 1)
            - begin: Time when the current spell began
            - spell.time: Time elapsed since the beginning of the current spell (R name)
            - spell_time: Alias of ``spell.time``
            - s.1, s.2, ...: Subsequence state **codes** (integers), not labels
            - transition: Boolean indicator (True if there's a state transition at this point)
            - Additional covariate columns (if covar was provided)

    Notes:
        **Void / missing:** Build ``SequenceData(..., void="%")`` (default) and include the
        void symbol in ``states`` if it appears in the data. ``seqsamm`` uses
        ``seqdata.void_code`` to drop subsequences containing void. Pass ``void=None`` only
        when you have no out-of-window padding. The auto-added Missing state is not void.

        **R parity check:** On the same toy data, compare ``nrow`` / ``len(samm.data)``,
        ``range(time)``, and ``spell.time`` summaries between R ``seqsamm()`` and this
        function (details in ``samm_examples.md``).

    Example:
        >>> # Suppose we have sequence data tracking employment states
        >>> # States: 'employed', 'unemployed', 'education'
        >>> # We want to analyze what happens in the next 3 time periods
        >>> samm_obj = sequence_analysis_multi_state_model(my_seqdata, sublength=3)
        >>> # Now we can use this for event history analysis
    """
    
    # Extract the sequence data as a numpy array (rows=individuals, columns=time points)
    # Each cell contains a numeric code representing a state (1, 2, 3, etc.)
    seqdata_array = seqdata.values
    n_individuals = seqdata_array.shape[0]  # Number of sequences/people
    n_timepoints = seqdata_array.shape[1]   # Length of each sequence

    if sublength < 2:
        raise ValueError(
            "sublength must be at least 2 because transition compares s.1 and s.2."
        )
    if n_timepoints <= sublength:
        raise ValueError(
            "Sequence length must be greater than sublength. "
            "R seqsamm uses time points 1:(L - sublength), so L must be larger than sublength."
        )

    # Create column names for the subsequence variables
    # For example, if sublength=3, this creates ['s.1', 's.2', 's.3']
    sname = [f's.{i+1}' for i in range(sublength)]
    
    # Get the IDs for each sequence
    # If the SequenceData has an ID column, use it; otherwise use row numbers
    if seqdata.id_col is not None:
        id_values = seqdata.ids
    else:
        id_values = np.arange(1, n_individuals + 1)
    
    # This will store all the person-period rows as we process each time point
    all_subseq_list = []
    
    # Track when each individual's current spell began
    # A "spell" is a continuous period in the same state
    # Initialize: everyone's spell begins at time 1
    spell_begin = np.ones(n_individuals, dtype=int)
    
    # R: for(tt in 1:(ncol(seqdata)-sublength)) — censoring limit L - sublength (not L - sublength + 1)
    void_codes = _void_codes_for_samm(seqdata)
    state_to_code = _state_to_code_map(seqdata)

    for tt in range(n_timepoints - sublength):
        subseq = seqdata_array[:, tt:(tt + sublength)]
        valid = _subseq_rows_valid(subseq, void_codes)

        transition = subseq[:, 0] != subseq[:, 1]

        if tt > 0:
            spell_reset_mask = seqdata_array[:, tt - 1] != seqdata_array[:, tt]
            spell_begin[spell_reset_mask] = tt + 1

        spell_time = (tt + 1) - spell_begin

        subseq_record = pd.DataFrame({
            "id": id_values,
            "time": tt + 1,
            "begin": spell_begin,
            SPELL_TIME_COL: spell_time,
            "transition": transition,
        })
        subseq_df = pd.DataFrame(subseq, columns=sname)
        subseq_record = pd.concat([subseq_record, subseq_df], axis=1)
        all_subseq_list.append(subseq_record.loc[valid].copy())

    result = pd.concat(all_subseq_list, ignore_index=True)

    if covar is not None:
        covar_indexed = _covar_rows_for_ids(covar, id_values)
        covar_indexed = covar_indexed.copy()
        covar_indexed["id"] = id_values
        result = result.merge(covar_indexed, on="id", how="left")

    result = result.sort_values(["id", "time"]).reset_index(drop=True)
    result[SPELL_TIME_ALIAS] = result[SPELL_TIME_COL]

    # Create and return the SAMM object
    samm_obj = SAMM(
        data=result,
        alphabet=seqdata.alphabet,
        labels=seqdata.labels,
        color_map=seqdata.color_map,
        sname=sname,
        sublength=sublength,
        state_to_code=state_to_code,
    )
    
    return samm_obj


def plot_samm(samm: SAMM, plot_type: str = "d", base_width: int = 15, 
              title: Optional[str] = None, save_as: Optional[str] = None, 
              dpi: int = 200, fontsize: int = 10):
    """
    Plot subsequences following transitions in the SAMM data using sequence index plots.
    
    R equivalent: **TraMineRextras** S3 method ``plot.SAMM(x, type = "d", ...)``, which
    internally calls **TraMineR** ``seqplot()`` on transition subsequences grouped by
    the state transitioned out of. This Python implementation does not call ``seqplot``;
    it uses matplotlib index-style subplots instead (see module docstring).
    
    **What does this show?**
    For each state, this displays the actual subsequence patterns (as colored bars) 
    that occur when individuals transition OUT of that state. Each row is one sequence,
    and colors represent different states in the subsequence.
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        plot_type (str): Type of plot to create (currently supports 'd' for sequence index plot)
        base_width (int): Base width for the figure. Default 15 (wider for better proportions).
        title (str, optional): Custom title for the plot
        save_as (str, optional): File path to save the plot (if None, plot is displayed)
        dpi (int): Resolution for saved images
        fontsize (int): Base font size for labels and titles
    
    Example:
        >>> samm_obj = sequence_analysis_multi_state_model(my_seqdata, sublength=3)
        >>> plot_samm(samm_obj, title="Transition Patterns")
    """
    
    # Import visualization utilities
    from io import BytesIO
    from sequenzo.visualization.utils import (
        create_standalone_legend, 
        combine_plot_with_legend,
        save_figure_to_buffer
    )
    from matplotlib.colors import ListedColormap
    
    # Filter to only rows where a transition occurs
    transition_rows = samm.data[samm.data['transition'] == True].copy()
    
    if len(transition_rows) == 0:
        print("No transitions found in the data.")
        return
    
    # Group by the starting state (s.1) to see transitions out of each state
    starting_states = sorted(transition_rows['s.1'].unique())
    
    # Create subplots: one for each starting state
    n_states = len(starting_states)
    ncols = min(3, n_states)  # Maximum 3 columns
    nrows = int(np.ceil(n_states / ncols))
    
    # Calculate dynamic heights for each subplot based on number of sequences
    # We'll use gridspec to allow different heights
    from matplotlib import gridspec
    
    # First, count sequences for each state to determine heights
    state_seq_counts = {}
    for state in starting_states:
        state_seq_counts[state] = len(transition_rows[transition_rows['s.1'] == state])
    
    # Calculate height ratios - base height per sequence, min 2.5, max 5 for better aspect ratio
    height_ratios = []
    for i in range(nrows):
        row_states = starting_states[i*ncols : (i+1)*ncols]
        if row_states:
            max_seqs_in_row = max([state_seq_counts[s] for s in row_states])
            # Height: 2.5-5 inches, scaled by number of sequences
            # Use smaller scaling factor (0.01 instead of 0.015) to make plots less stretched
            height = min(5, max(2.5, max_seqs_in_row * 0.01))
            height_ratios.append(height)
    
    # Calculate total figure height with more spacing
    total_height = sum(height_ratios) + (nrows - 1) * 2.0  # Add more spacing between rows
    
    # Create figure with GridSpec for flexible heights
    fig = plt.figure(figsize=(base_width, total_height))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=height_ratios,
                          hspace=0.5, wspace=0.25)  # Adjusted spacing for better layout
    
    # Contiguous 0..K-1 indices for imshow (state codes may not be 1..K)
    plot_codes = sorted(int(c) for c in samm.state_to_code.values())
    code_to_idx = {code: i for i, code in enumerate(plot_codes)}
    cmap_colors = []
    for code in plot_codes:
        if code in samm.color_map:
            cmap_colors.append(samm.color_map[code])
        elif code - 1 in samm.color_map:
            cmap_colors.append(samm.color_map[code - 1])
        else:
            cmap_colors.append("#cccccc")
    cmap = ListedColormap(cmap_colors)
    imshow_vmin, imshow_vmax = 0, max(0, len(plot_codes) - 1)

    def _code_display_name(code) -> str:
        for state, sc in samm.state_to_code.items():
            if sc == code:
                idx_lab = samm.alphabet.index(state) if state in samm.alphabet else None
                return samm.labels[idx_lab] if idx_lab is not None else str(state)
        return str(code)

    # For each starting state, create a sequence index plot
    for idx, state_code in enumerate(starting_states):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        state_data = transition_rows[transition_rows["s.1"] == state_code].copy()
        subseq_matrix = state_data[samm.sname].to_numpy(dtype=float)
        numeric_matrix = subseq_matrix.astype(float)
        plot_matrix = _remap_codes_for_imshow(numeric_matrix, code_to_idx)

        # Plot with masked array for NaN handling
        ax.imshow(
            np.ma.masked_invalid(plot_matrix),
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=imshow_vmin,
            vmax=imshow_vmax,
        )
        
        # Disable grid
        ax.grid(False)
        
        # Set title showing the starting state with count
        num_seqs = numeric_matrix.shape[0]
        title_text = f"Transitions out of: {_code_display_name(state_code)} (n={num_seqs})"
        
        # Break long titles into multiple lines
        if len(title_text) > 35:  # If title is too long
            # Try to break at a natural point
            if 'Transitions out of:' in title_text:
                parts = title_text.split('Transitions out of:')
                if len(parts) == 2:
                    title_text = f'Transitions out of:\n{parts[1].strip()}'
        
        ax.set_title(title_text, fontsize=fontsize+1, pad=12, color='black')
        
        # X-axis: time steps in subsequence
        ax.set_xlabel('Subsequence Position', fontsize=fontsize, labelpad=8, color='black')
        xticks = np.arange(len(samm.sname))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"t+{i}" for i in range(len(samm.sname))],
                          fontsize=fontsize-2, color='gray')
        
        # Y-axis: sequence count
        ax.set_ylabel('Sequences', fontsize=fontsize, labelpad=8, color='black')
        
        # Smart y-tick display based on sequence count
        if num_seqs <= 10:
            yticks = np.arange(num_seqs)
            ax.set_yticks(yticks)
            ax.set_yticklabels(range(1, num_seqs + 1), fontsize=fontsize-2, color='gray')
        elif num_seqs <= 50:
            # Show every 5th or 10th
            step = 5 if num_seqs <= 25 else 10
            yticks = np.arange(0, num_seqs, step)
            if yticks[-1] != num_seqs - 1:
                yticks = np.append(yticks, num_seqs - 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y + 1) for y in yticks], fontsize=fontsize-2, color='gray')
        else:
            # Show quartiles for large numbers
            ytick_positions = [0, num_seqs // 4, num_seqs // 2, 3 * num_seqs // 4, num_seqs - 1]
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels([str(pos + 1) for pos in ytick_positions], 
                              fontsize=fontsize-2, color='gray')
        
        # Style axis spines and ticks like index plot
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(0.8)
        
        # Tick parameters matching index plot style
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')
        ax.tick_params(axis='both', which='major', direction='out')
    
    # Adjust layout first
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave less space at top for title
    
    # Add overall title if provided (after tight_layout to prevent overlap)
    if title:
        fig.suptitle(title, fontsize=fontsize+4, y=0.93, color='black')
    
    # Save main figure to buffer
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)
    
    # Create standalone legend using the same style as index plot
    colors = {}
    for i, state in enumerate(samm.alphabet):
        code = int(samm.state_to_code[state])
        label = samm.labels[i] if i < len(samm.labels) else str(state)
        colors[label] = samm.color_map.get(code, samm.color_map.get(i + 1, "#cccccc"))
    legend_buffer = create_standalone_legend(
        colors=colors,
        labels=samm.labels,
        ncol=min(5, len(samm.labels)),
        figsize=(base_width, 1),
        fontsize=fontsize,
        dpi=dpi
    )
    
    # Combine plot with legend
    if save_as and not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        save_as = save_as + '.png'
    
    combined_img = combine_plot_with_legend(
        main_buffer,
        legend_buffer,
        output_path=save_as,
        dpi=dpi,
        padding=20
    )
    
    # Display combined image
    plt.figure(figsize=(base_width, total_height + 1))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.show()
    plt.close('all')


def seqsammseq(samm: SAMM, spell: str) -> pd.DataFrame:
    """
    Extract subsequences that follow a specific state (spell).
    
    R equivalent: **TraMineRextras** ``seqsammseq(samm, spell)``, which returns an
    ``stslist`` sequence object (TraMineR ``seqdef``-style). This Python version returns
    a ``DataFrame`` of subsequence columns (``s.1``, ``s.2``, ...) for transition rows
    where ``s.1 == spell``.
    
    This function returns all the subsequences that occur after a given state,
    specifically when there is a transition OUT of that state.
    
    **Why is this useful?**
    It helps you analyze what happens after a particular state. For example,
    if you're studying employment sequences, you might want to know:
    "What happens after someone becomes unemployed?" or
    "What patterns follow graduation?"
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state you want to analyze transitions from
                     (e.g., 'employed', 'single', 'education')
    
    Returns:
        pd.DataFrame: A DataFrame containing only the subsequence columns (s.1, s.2, ...)
                     for rows where:
                     1. The starting state (s.1) matches the specified spell
                     2. A transition occurs at that point
    
    Example:
        >>> # Get all subsequences following unemployment
        >>> unemployed_subsequences = seqsammseq(samm_obj, spell='unemployed')
        >>> print(unemployed_subsequences.head())
        # This shows what typically happens after someone becomes unemployed
    """
    
    spell_code = _resolve_spell_value(spell, samm.state_to_code, samm.labels, samm.alphabet)
    condition = (samm.data["s.1"] == spell_code) & (samm.data["transition"].astype(bool))
    
    # Extract only the subsequence columns
    subsequences = samm.data.loc[condition, samm.sname].copy()
    
    # Reset index for cleaner output
    subsequences = subsequences.reset_index(drop=True)
    
    return subsequences


def _expand_typology_for_transitions(
    samm: SAMM,
    spell: Union[str, int, float],
    mapping: Union[Dict, pd.Series],
    by: Optional[str] = None,
    cluster_to_name: Optional[Dict] = None,
) -> np.ndarray:
    """
    Build a row-aligned typology vector for transition rows given a mapping.

    Parameters
    ----------
    samm : SAMM
        The SAMM object.
    spell : str
        The state to analyze transitions out of.
    mapping : dict or pandas.Series
        Either a mapping of id -> cluster/label, or (id, begin) -> cluster/label.
        Values can be final label strings, or cluster ids to be mapped via cluster_to_name.
    by : {"id", "id_begin"}, optional
        If None, auto-detect by inspecting mapping keys/index. Use "id_begin" when
        mapping is keyed by (id, begin).
    cluster_to_name : dict, optional
        Mapping from cluster id to human-readable label. Required if mapping values
        are cluster ids rather than label strings.

    Returns
    -------
    numpy.ndarray
        A vector of labels aligned to samm.data.loc[(s.1==spell) & transition].
    """
    spell_code = _resolve_spell_value(spell, samm.state_to_code, samm.labels, samm.alphabet)
    condition = (samm.data["s.1"] == spell_code) & (samm.data["transition"].astype(bool))
    trans_df = samm.data.loc[condition, ["id", "begin"]].copy()

    # Normalize mapping to a dict for fast lookup
    if isinstance(mapping, pd.Series):
        if mapping.index.nlevels == 1:
            normalized: Dict = mapping.to_dict()
            inferred_by = 'id'
        elif mapping.index.nlevels == 2:
            normalized = {tuple(idx): val for idx, val in mapping.items()}
            inferred_by = 'id_begin'
        else:
            raise ValueError("Mapping Series index must be 1 or 2 levels: id or (id, begin)")
    else:
        normalized = dict(mapping)
        # Auto-detect key type when by is not provided
        if by is None:
            if len(normalized) == 0:
                inferred_by = 'id'  # default
            else:
                sample_key = next(iter(normalized.keys()))
                inferred_by = 'id_begin' if isinstance(sample_key, tuple) and len(sample_key) == 2 else 'id'
        else:
            inferred_by = by

    labels: List[str] = []
    missing_keys: List[Union[int, Tuple[int, int]]] = []

    if inferred_by == 'id':
        for pid in trans_df['id'].tolist():
            if pid not in normalized:
                missing_keys.append(pid)
                labels.append(None)
                continue
            val = normalized[pid]
            # If val is numeric-like and cluster_to_name is provided, map to name
            if cluster_to_name is not None and pd.notna(val):
                try:
                    labels.append(cluster_to_name[val])
                except KeyError:
                    raise ValueError(f"cluster_to_name is missing key {val!r} for id {pid}")
            else:
                labels.append(val)
    elif inferred_by == 'id_begin':
        ids = trans_df['id'].to_list()
        begins = trans_df['begin'].to_list()
        for pid, b in zip(ids, begins):
            key = (pid, b)
            if key not in normalized:
                missing_keys.append(key)
                labels.append(None)
                continue
            val = normalized[key]
            if cluster_to_name is not None and pd.notna(val):
                try:
                    labels.append(cluster_to_name[val])
                except KeyError:
                    raise ValueError(f"cluster_to_name is missing key {val!r} for (id, begin) {key}")
            else:
                labels.append(val)
    else:
        raise ValueError("Parameter 'by' must be one of {'id', 'id_begin'}")

    if missing_keys:
        sample = missing_keys[:5]
        raise ValueError(
            f"Missing {len(missing_keys)} keys in mapping for transitions from '{spell}'. "
            f"Examples: {sample}. You can provide (id, begin) or id mappings, "
            f"and use cluster_to_name to map cluster ids to names."
        )

    return np.asarray(labels, dtype=object)


def set_typology(
    samm: SAMM,
    spell: str,
    typology: Union[pd.Series, np.ndarray, list, None] = None,
    *,
    clusters: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cluster_to_name: Optional[Dict] = None,
    mapping: Optional[Union[Dict, pd.Series]] = None,
    by: Optional[str] = None
) -> SAMM:
    """
    Assign a typology classification to subsequences following a specific state.
    
    **R note:** TraMineRextras documents typology assignment inside ``seqsammeha()``;
    the source also defines a placeholder ``typology<-`` setter that does not implement
    the full logic. ``set_typology()`` is a Sequenzo helper so typologies can be set
    before calling ``seqsammeha()`` (with ``clusters``, ``mapping``, etc.).
    
    This function allows you to categorize the different patterns that occur
    after transitioning out of a particular state. This is useful for creating
    meaningful groups for further analysis.
    
    **What is a typology?**
    A typology is a classification system. For example, after unemployment,
    you might classify subsequences as:
    - "Quick reemployment" (gets job within 3 months)
    - "Long-term unemployment" (stays unemployed > 6 months)
    - "Exit labor force" (moves to education or retirement)
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state for which you're setting typologies
        typology (array-like, optional): Final labels for each transition row (length = n_transitions).
        clusters (array-like, optional): Cluster ids per transition row (length = n_transitions).
        cluster_to_name (dict, optional): Mapping from cluster id -> label name. Used with clusters
                                          or when mapping values are cluster ids.
        mapping (dict or pandas.Series, optional): id -> cluster/label or (id, begin) -> cluster/label.
        by (str, optional): 'id' or 'id_begin'. If None, auto-detect from mapping keys.
    
    Returns:
        SAMM: The updated SAMM object with typology column filled in
    
    Example:
        >>> # First, identify transitions from unemployment
        >>> unemployed_transitions = (samm_obj.data['s.1'] == 'unemployed') & samm_obj.data['transition']
        >>> # Create your typology based on some logic
        >>> my_typology = ['quick_return', 'education', 'long_term', ...]  # One label per transition
        >>> # Apply the typology
        >>> samm_obj = set_typology(samm_obj, spell='unemployed', typology=my_typology)
    """
    
    spell_code = _resolve_spell_value(spell, samm.state_to_code, samm.labels, samm.alphabet)
    condition = (samm.data["s.1"] == spell_code) & (samm.data["transition"].astype(bool))

    n_transitions = int(condition.sum())

    labels_array: Optional[np.ndarray] = None

    # Case 1: direct typology vector
    if typology is not None:
        if isinstance(typology, pd.Series):
            labels_array = typology.values
        else:
            labels_array = np.asarray(typology, dtype=object)
        if len(labels_array) != n_transitions:
            raise ValueError(
                f"Length mismatch: provided length {len(labels_array)} but there are {n_transitions} "
                f"transitions from state '{spell}'. You should provide a typology vector of length n_transitions "
                f"(one label per transition row), not a list of unique type names. Use clusters+cluster_to_name "
                f"or mapping parameters instead."
            )

    # Case 2: clusters aligned to transition rows + mapping dict
    elif clusters is not None:
        clusters_array = clusters.values if isinstance(clusters, pd.Series) else np.asarray(clusters)
        if len(clusters_array) != n_transitions:
            raise ValueError(
                f"Length mismatch: clusters length {len(clusters_array)} must match n_transitions={n_transitions}"
            )
        if cluster_to_name is not None:
            try:
                labels_array = np.asarray([cluster_to_name[c] for c in clusters_array], dtype=object)
            except KeyError as e:
                raise ValueError(f"cluster_to_name is missing key {e.args[0]!r}")
        else:
            # Assume clusters are already label strings
            labels_array = clusters_array.astype(object)

    # Case 3: mapping keyed by id or (id, begin)
    elif mapping is not None:
        labels_array = _expand_typology_for_transitions(
            samm=samm, spell=spell, mapping=mapping, by=by, cluster_to_name=cluster_to_name
        )

    else:
        raise ValueError(
            "You must provide one of: typology (row-aligned), clusters+cluster_to_name (row-aligned), "
            "or mapping (id or (id, begin) to cluster/label)."
        )

    # Assign the typology labels to the corresponding rows
    samm.data.loc[condition, 'typology'] = labels_array
    
    return samm


def seqsammeha(
    samm: SAMM,
    spell: str,
    typology: Union[pd.Series, np.ndarray, list, None] = None,
    *,
    clusters: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cluster_to_name: Optional[Dict] = None,
    mapping: Optional[Union[Dict, pd.Series]] = None,
    by: Optional[str] = None,
    persper: bool = True
) -> pd.DataFrame:
    """
    Generate a dataset for Event History Analysis (EHA) with typology outcomes.
    
    R equivalent: **TraMineRextras** ``seqsammeha(samm, spell, typology, persper = TRUE)``.
    Adds ``SAMMtypology``, ``lastobs``, and dummy columns ``SAMM<typology_label>`` (R uses
    the same ``SAMM`` prefix on typology level names). In R, ``typology`` is a vector with
    one label per transition out of ``spell``; this Python API also accepts ``clusters``,
    ``mapping``, and ``cluster_to_name`` via :func:`set_typology`.
    
    This function prepares your data for statistical models (like logistic regression
    or survival analysis) that estimate the probability of different outcomes
    following a specific state.
    
    **What is Event History Analysis?**
    EHA examines the timing and nature of events. For example:
    - "What factors predict returning to work after unemployment?"
    - "How long do people stay in education before entering the labor force?"
    
    **Person-period vs. Spell-level data:**
    - person-period (persper=True): One row for EACH time point in the spell
      Good for: Time-varying effects, duration dependence
    - spell-level (persper=False): One row per spell (only the last observation)
      Good for: Simpler models, overall spell outcomes
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state you're analyzing (e.g., 'unemployed', 'single')
        typology (array-like, optional): Final labels for each transition row (length = n_transitions)
        clusters (array-like, optional): Cluster ids per transition row (length = n_transitions)
        cluster_to_name (dict, optional): Mapping from cluster id -> label name
        mapping (dict or pandas.Series, optional): id -> cluster/label or (id, begin) -> cluster/label
        by (str, optional): 'id' or 'id_begin'. If None, auto-detect
        persper (bool): If True, return person-period data (multiple rows per spell).
                        If False, return spell-level data (one row per spell).
    
    Returns:
        pd.DataFrame: A dataset ready for event history analysis with:
            - All original SAMM variables (id, time, spell_time, etc.)
            - SAMMtypology: The typology classification (with "None" for non-events)
            - lastobs: Boolean indicating if this is the last observation of a spell
            - SAMM[type1], SAMM[type2], ...: Binary indicators for each typology category
              (these are your outcome variables for analysis)
    
    Example:
        >>> # Define typologies for transitions from unemployment
        >>> typology = ['reemployed', 'education', 'reemployed', 'retired', ...]
        >>> # Create EHA dataset
        >>> eha_data = seqsammeha(samm_obj, spell='unemployed', typology=typology, persper=True)
        >>> # Now you can use this with logistic regression, Cox models, etc.
        >>> # For example: predict probability of reemployment vs. other outcomes
    """
    
    spell_code = _resolve_spell_value(spell, samm.state_to_code, samm.labels, samm.alphabet)
    labels_array = _build_typology_labels(
        samm,
        spell_code,
        typology=typology,
        clusters=clusters,
        cluster_to_name=cluster_to_name,
        mapping=mapping,
        by=by,
    )
    types = _typology_type_levels(typology, labels_array)

    spell_condition = samm.data["s.1"] == spell_code
    ppdata = samm.data.loc[spell_condition].copy()

    ppdata["SAMMtypology"] = "None"
    trans_mask = ppdata["transition"].astype(bool)
    ppdata.loc[trans_mask, "SAMMtypology"] = labels_array

    spell_col = SPELL_TIME_COL if SPELL_TIME_COL in ppdata.columns else SPELL_TIME_ALIAS
    ppdata["lastobs"] = (
        ppdata.groupby(["id", "begin"])[spell_col].transform("max") == ppdata[spell_col]
    )

    for type_label in types:
        col_name = f"SAMM{type_label}"
        ppdata[col_name] = (ppdata["SAMMtypology"] == type_label).astype(int)
    
    # If persper=False, return only the last observation of each spell
    if not persper:
        ppdata = ppdata[ppdata['lastobs']].copy()
    
    # Reset index for clean output
    ppdata = ppdata.reset_index(drop=True)
    
    return ppdata


# Define what gets imported with "from module import *"
__all__ = [
    'SAMM',
    'sequence_analysis_multi_state_model',
    'plot_samm',
    'seqsammseq',
    'set_typology',
    'seqsammeha',
    '_expand_typology_for_transitions',
    # Keep old names for backward compatibility
    'seqsamm'
]

# Backward compatibility: same name as TraMineRextras seqsamm()
seqsamm = sequence_analysis_multi_state_model
