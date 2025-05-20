"""
@Author  : 李欣怡
@File    : get_LCP_length_for_2_seq.py
@Time    : 2025/5/20 11:25
@Desc    : 
"""

from sequenzo.define_sequence_data import SequenceData

def get_LCP_length_for_2_seq(data1 = None, data2 = None, id1 = None, id2 = None):
    # Check parameters is None
    if data1 is None or data2 is None:
        raise ValueError("[!] 'data1' and 'data2' must be set.")

    if id1 is None or id2 is None:
        raise ValueError("[!] 'id1' and 'id2' must be set.")

    # Check type
    if not isinstance(data1, SequenceData) or not isinstance(data2, SequenceData):
        raise TypeError("[!] sequences must be sequence objects")

    if not isinstance(id1, int) or not isinstance(id2, int):
        raise TypeError("[!] 'id1' and 'id2' must be int.")

    # Check id
    if id1 > data1.seqdata.shape[0] or id2 > data2.seqdata.shape[0] or id1 < 0 or id2 < 0:
        raise ValueError("[!] 'data1' or 'data2' has no such id.")

    # Check states
    if len(data1.states) != len(data2.states) or any(a != b for a, b in zip(data1.states, data2.states)):
        raise ValueError("[!] The alphabet of both sequences have to be same.")

    # Get the two sequences which are compared
    seq1 = data1.seqdata.iloc[id1].to_numpy()
    seq2 = data2.seqdata.iloc[id2].to_numpy()

    boundary = min(len(seq1), len(seq2))

    # Compute LCP length
    length = 0
    while seq1[length] == seq2[length] and length < boundary:
        length += 1

    return length