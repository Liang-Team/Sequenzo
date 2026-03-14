#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Module for random generator structured sequence.

A sequence is an ordered set of items: it does not generate itemsets neither generate timestamps.

Both patterns and sequences are randomly generated see module db_generator for more details

An item in a sequence can be contribute to several (different) pattern occurrences.

.. ::see db_generator
"""

__author__ = "Thomas Guyet"
__copyright__ = "Copyright 2019, AGROCAMPUS-OUEST/IRISA"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "Thomas Guyet"
__email__ = "thomas.guyet@irisa.fr"

import numpy as np
import sys, getopt
import warnings
import math

from db_generator import *

class puresequence(sequence):
    """Representation of a sequence holding only purely sequential patterns (no timestamps)
    
    See also
    --------
    db_generator.pattern
    """
    def __init__(self, rl=20, fl=False):
        """
        :param rl: mean/fixed length of the sequence
        :param fl: set a fixed length (exactly rl)
        """
        self.fixedlength=fl
        super().__init__(rl,0)
       
    def self_generate(self, item_gen):
        """ Generate the sequence of items from the patterns it contains and according to the require length
        
        :param item_gen: random itemset generator used to mix the pattern occurrences with random items
        """
            
        # no patterns inside: fully random sequences
        if len(self.patterns)==0:
            if self.fixedlength:
                l=self.requiredlen
            else:
                l=int(np.random.normal(self.requiredlen, self.requiredlen/10))
            for i in range(l):
                item = item_gen.generate()
                self.seq.append( (item,len(self.seq)) )
            return

        # sequence that must include patterns
        readpos = np.zeros( len(self.patterns) ) #retient la position de lecture du motif
        available=list(range(0, len(self.patterns)))
        hurryness=[]
        for pat in self.patterns:
            if pat.maxgap==math.inf:
                hurryness.append(math.inf)
            else:
                hurryness.append(self.requiredlen-len(pat.sequence))
        hurryness= np.array(hurryness,dtype='float')
            
        self.seq=[]
        lastpat=None
        while( len(available)>0 ):
        
            mh=np.min(hurryness)
            if mh<=0 :
                pat = int(np.random.choice(np.where( hurryness==mh )[0], 1)) #select one of the patterns whose hurryness is the smaller
            else:
                in_a_hurry=False
                for level in range( self.requiredlen ):
                    if np.sum(hurryness<=level) > level:
                        in_a_hurry=True
                        #select one of the patterns that imposes a constraint
                        pat = int(np.random.choice(np.where( hurryness<=level )[0], 1))
                        break
                
                if not in_a_hurry:
                    #generate a random item
                    if np.random.random()<float(self.requiredlen-self.__patcontentlen__())/float(self.requiredlen):
                        item = item_gen.generate()
                        self.seq.append( (item,len(self.seq)) )
                        #one item has been added: the hurryness of all patterns are 
                        
                        hurryness= np.array([h-1 for h in hurryness],dtype='float')
                        
                        #go directly to the generation of the next item
                        continue
                        
                    #select randomly one pattern
                    pat=available[np.random.randint(0,len(available))]
            
            #HERE: a element of the pattern 'pat' has to be pushed into the current sequence
            #add its current item to the sequence
            pos = int(readpos[pat])
            
            item = self.patterns[pat].sequence[pos]
            if lastpat==pat or len(self.seq)==0 or item != self.seq[len(self.seq)-1][0] :
                """
                if the item is similar to the last inserted item
                (from another pattern), then it is not added (merge patterns)
                """
                self.seq.append( (item,len(self.seq)) )
            
            #update the hurryness of the pattern: 
            hurryness= np.array([h-1 for h in hurryness],dtype='float')     # decrease the hurryness
            # the modified pattern is granted with the pat.maxgap hurryness (the max number of step before using the next item)
            if not self.patterns[pat].maxgap is None and hurryness[pat]!=math.inf:           
                hurryness[pat]=self.patterns[pat].maxgap
            
            #update the pointer of the current item of the selected pattern
            pos += 1
            readpos[pat] = pos
            #if it is the end of the patterns, it is removed from the available patterns, and hurryness is switched to infinity
            if pos==len(self.patterns[pat]):
                available.remove(pat)
                hurryness[pat]=math.inf
            lastpat=pat
            
            
        #add random items at the end
        if self.fixedlength:
            if len(self.seq)>self.requiredlen:
                warnings.warn("Generated sequence has a length above the fixed sequence length: too much constraints!")
            while len(self.seq)<self.requiredlen:
                item = item_gen.generate()
                self.seq.append( (item,len(self.seq)) )
        else:
            while np.random.random()<float(self.requiredlen-self.__patcontentlen__())/float(self.requiredlen):
                item = item_gen.generate()
                self.seq.append( (item,len(self.seq)) )

class puresequence_generator(sequence_generator):
    """Sequence factory for sequences with purely sequential patterns
    """
    def __init__(self,fl=False):
        super().__init__()
        self.fl=fl
    
    def generate(self, l):
        return puresequence(l,self.fl)


class seqdb_generator(db_generator):
    """Generator of a dataset containing sequences with purely sequential patterns
    
    See also
    --------
    puresequence_generator
    """

    def __init__(self, n=None, mg=None, lp=4, fixedlength=False):
        """Constructor of the db generator
        
        :param n: vocabulary size (default 100)
        :param mg: maxgaps in patterns (default none)
        :param lp: mean pattern length
        """
        if not n is None:
            self.n=n
        else:
            self.n = 100     # number of items
        self.fl="uniform"    # item frequency distribution
        
        
        itemgen= item_generator(n=self.n, fl=self.fl)
        seqgen = puresequence_generator(fixedlength)
        patgen = pattern_generator(itemgen, mg=mg)
        pattern_generator.lpat=lp #change class attribute
        
        super().__init__(itemgen, seqgen, patgen)

if __name__ == "__main__":
    generator=seqdb_generator(n=20, mg=None, lp=4, fixedlength=False)
    
    sequences = generator.generate(nb=10, l=7, npat=3, th=.2)

    print("======== PATTERNS =======")
    print(generator.output_patterns())
    
    print("======== SEQUENCES =======")
    for s in sequences:
        print(str(s))
    
