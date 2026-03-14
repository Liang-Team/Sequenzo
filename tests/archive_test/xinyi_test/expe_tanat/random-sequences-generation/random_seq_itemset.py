#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Random sequence generator.

It includes simple random generators of sequence of items or itemsets (not based on patterns).

"""

__author__ = "Thomas Guyet"
__copyright__ = "Copyright 2019, AGROCAMPUS-OUEST/IRISA"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "Thomas Guyet"
__email__ = "thomas.guyet@irisa.fr"
    

import seqdb_generator as seqgen
import numpy as np


def _seq_to_tuple(s):
    """Convert sequence to hashable tuple for deduplication."""
    return tuple((item, pos) for item, pos in s.seq)


def _copy_sequence(orig):
    """Create a copy of a sequence (new sequence object with same content)."""
    s = seqgen.sequence(rl=orig.requiredlen, d=orig.duration)
    s.seq = [(item, pos) for item, pos in orig.seq]
    return s


def gen_seqdb(seqnb, length, maxitems, fixedlength, uniqueness_rate=100):
    """Fully random sequence generation. Generate sequences of itemsets.

    :param seqnb: number of sequences
    :param length: mean length of the sequences
    :param maxitems: size of the vocabulary (maximum number of items)
    :param fixedlength: if True, all sequences have fixed length
    :param uniqueness_rate: percentage (0-100) of sequences that are unique (each appears once).
                            U=5 means 5%% of sequences are unique, 95%% are duplicates.
                            U=100 (default) means all sequences are unique.
    :return: generated sequences
    """
    U = max(0, min(100, uniqueness_rate))
    num_unique = int(seqnb * U / 100)
    num_repeated = seqnb - num_unique

    sequences = []
    # Use uniform distribution to reduce collision (gaussian makes many similar sequences)
    item_gen = seqgen.item_generator(maxitems, fl="uniform")
    seen = set()  # for deduplication
    max_attempts = max(5000, 100 * num_unique)

    def _generate_one(rng=None):
        """Generate one random sequence. rng=None uses global numpy random."""
        s = seqgen.sequence()
        if fixedlength:
            rlength = length
        else:
            norm = rng.normal(length, max(1, int(length / 10))) if rng is not None else np.random.normal(length, max(1, int(length / 10)))
            rlength = int(norm)
        rlength = max(1, rlength)
        for j in range(rlength):
            item = rng.randint(0, maxitems) if rng is not None else item_gen.generate()
            s.seq.append((item, j))
        return s

    # Phase 1: try random generation
    attempts = 0
    while len(sequences) < num_unique and attempts < max_attempts:
        s = _generate_one()
        key = _seq_to_tuple(s)
        if key not in seen:
            seen.add(key)
            sequences.append(s)
        attempts += 1

    # Phase 2: deterministic fallback - guarantees uniqueness via seeded RNG
    if len(sequences) < num_unique:
        seed_offset = 0
        while len(sequences) < num_unique:
            for i in range(seed_offset, seed_offset + (num_unique - len(sequences)) * 2):
                if len(sequences) >= num_unique:
                    break
                rng = np.random.RandomState(i)
                s = seqgen.sequence()
                rlength = length if fixedlength else max(1, int(rng.normal(length, max(1, int(length / 10)))))
                for j in range(rlength):
                    s.seq.append((rng.randint(0, maxitems), j))
                key = _seq_to_tuple(s)
                if key not in seen:
                    seen.add(key)
                    sequences.append(s)
            seed_offset += 100000
            if len(sequences) >= num_unique:
                break

    # Generate template for repeated part and add num_repeated copies
    if num_repeated > 0:
        repeated_template = None
        attempts = 0
        while repeated_template is None and attempts < max_attempts:
            s = _generate_one()
            if _seq_to_tuple(s) not in seen:
                repeated_template = s
            attempts += 1
        if repeated_template is None:
            repeated_template = _generate_one()  # fallback: use any sequence
        for _ in range(num_repeated):
            sequences.append(_copy_sequence(repeated_template))

    # Shuffle so unique and repeated are mixed
    np.random.shuffle(sequences)
    return sequences
    
def gen_seq_itemset():

    fout = open(outputfile, "w")
    for i in range(seqnb):
        if i!=0:
            fout.write("\n")
        if fixedlength:
            rlength=length
        else:
            rlength=int(np.random.normal(length,max(1,int(length/10))))
        for j in range(rlength):
            alreadyAdded=[]
            itemset=[]
            for k in range(itemsetSize):
                item = np.random.randint(0,maxitems+1,1)[0]
                while item in alreadyAdded:
                    item = np.random.randint(0,maxitems+1,1)[0]
                alreadyAdded.append(item)
                itemset.append(item)
            itemset.sort()
            for it in itemset[:-1]:
                if asp:
                    fout.write( "seq("+str(i) +","+str(j) +","+str(it) +"). ")
                else:
                    fout.write( str(it) + ":")
            if asp:
                fout.write( "seq("+str(i) +","+str(j) +","+str(itemset[len(itemset)-1]) +"). ")
            else:
                fout.write( str(itemset[len(itemset)-1]) )
            if j!=(rlength-1) and not asp:
                fout.write(",")
    fout.close()
