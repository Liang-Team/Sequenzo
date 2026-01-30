"""
Central hub for building suffix trees: position vs spell mode.

Analogous to get_distance_matrix which dispatches to RLCP / RLCPspell based on method,
this module provides a single entry point that dispatches to position-based or
spell-based suffix tree based on `mode`.

Usage:
    # Position-based (default, same as before): level = time index from end
    sequences, states = convert_to_suffix_tree_data(df, time_cols)
    tree = build_suffix_tree(sequences)
    # or explicitly:
    tree = build_suffix_tree(sequences, mode="position")

    # Spell-based: level = spell index from end (aligned with RLCPspell distance)
    seqdata = SequenceData(df, time=time_cols, id_col="id", states=states)
    tree = build_suffix_tree(seqdata, mode="spell", expcost=0)
    # expcost=0: ignore duration; expcost>0: duration-weighted indicators

@Author  : Yuqi Liang 梁彧祺
@File    : hub.py
@Time    : 2026/1/29 23:17
@Desc    : Central dispatch for suffix tree (position / spell).
"""
from typing import Union, List, Any

from sequenzo.define_sequence_data import SequenceData

from .system_level_indicators import SuffixTree, _build_suffix_tree_position
from .spell_level_indicators import build_spell_suffix_tree, SpellSuffixTree


def build_suffix_tree(
    data: Union[List[List[Any]], SequenceData],
    mode: str = "position",
    expcost: float = 0.0,
) -> Union[SuffixTree, SpellSuffixTree]:
    """
    Build a suffix tree (central hub).

    Parameters
    ----------
    data : list of sequences, or SequenceData
        - For mode="position": list of lists, each inner list = states at consecutive
          time points (e.g. from convert_to_suffix_tree_data(df, time_cols)).
          Can also pass SequenceData; sequences will be extracted from it.
        - For mode="spell": must be SequenceData. Spell representation is computed
          internally (DSS + duration).
    mode : str, default "position"
        - "position": level = time index from end (same as original suffix tree).
          Aligns with RLCP distance (position-wise from end).
        - "spell": level = spell index from end (last spell, last two spells, ...).
          Aligns with RLCPspell distance (spell-wise from end).
    expcost : float, default 0.0
        Only used when mode="spell".
        - 0: structure and indicators ignore duration (state-only, like RLCPspell expcost=0).
        - >0: duration influences derived indicators (e.g. JS convergence uses
          spell-length weighting). Larger expcost = longer spells matter more.

    Returns
    -------
    SuffixTree or SpellSuffixTree
        - mode="position" -> SuffixTree (original)
        - mode="spell" -> SpellSuffixTree (new)

    Examples
    --------
    >>> from sequenzo import SequenceData, build_suffix_tree, convert_to_suffix_tree_data
    >>>
    >>> # Position mode (original behavior)
    >>> seqs, states = convert_to_suffix_tree_data(df, ["C1", "C2", "C3"])
    >>> tree = build_suffix_tree(seqs)
    >>>
    >>> # Spell mode (aligned with RLCPspell)
    >>> seqdata = SequenceData(df, time=["C1","C2","C3"], id_col="id", states=states)
    >>> tree = build_suffix_tree(seqdata, mode="spell", expcost=0.5)
    """
    mode_lower = (mode or "position").strip().lower()
    if mode_lower not in ("position", "spell"):
        raise ValueError(
            f"[!] mode must be 'position' or 'spell', got '{mode}'.\n"
            "    - position: level = time index from end (original suffix tree)\n"
            "    - spell: level = spell index from end (aligned with RLCPspell distance)"
        )

    if mode_lower == "position":
        if isinstance(data, SequenceData):
            # Extract position-based sequences from SequenceData (map indices to state labels)
            vals = data.seqdata[data.time].values
            state_list = list(data.states)
            sequences = [
                [state_list[int(v)] if 0 <= int(v) < len(state_list) else v for v in row]
                for row in vals.tolist()
            ]
        elif isinstance(data, (list, tuple)) and data and hasattr(data[0], "__iter__"):
            sequences = data
        else:
            raise TypeError(
                "[!] For mode='position', data must be a list of sequences "
                "(list of lists) or SequenceData."
            )
        return _build_suffix_tree_position(sequences)

    # mode == "spell"
    if not isinstance(data, SequenceData):
        raise TypeError(
            "[!] For mode='spell', data must be SequenceData.\n"
            "    Use: SequenceData(df, time=..., id_col=..., states=...)\n"
            "    Then: build_suffix_tree(seqdata, mode='spell', expcost=0)"
        )
    if expcost < 0:
        raise ValueError("[!] expcost must be non-negative (use 0 to ignore duration).")

    return build_spell_suffix_tree(data, expcost=expcost)
