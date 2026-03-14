#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sequence database for discriminant generation
TO finsh
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
import scipy.sparse.csgraph
        



class discr_db_generator:
    """Databases generator for discriminant pattern extraction
    Generate to databases pos and negatives
    """

    def __init__(self, n=None):
        """create factories of items and constraints
        """
        self.patgen = None      # last generated pattern generator
        self.patterns = []      # set of patterns
        self.db_pos = []        # database of sequences
        self.db_neg = []        # database of sequences
        
        self.nbex = 1000        # number of sequences in the database
        self.l = 30             # mean length of the sequences
        self.n = 100            # number of items
        self.fl="uniform"       # item frequency law of distribution
        self.nbpat = 5          # number of the patterns
        self.lpat = 5           # mean length of the patterns
        self.th = 0.10          # pattern threshold
        self.gr = 1.50          # growth rate (represent the growth rate of positive wrt negatives)
        self.dc = 0.30          # constraint density (if 0: no constraints, if 1 each pair of event are temporaly constraints)
    
        if not n is None:
            self.n=n
            
        # create an item factory
        self.itemgen = item_generator(n=self.n, fl=self.fl)
        # create a constraint factory
        self.constgen = constraint_generator(-100,100,0.1,200)
        #chronicle factory
        self.chrogen = chronicle_generator(self.itemgen, self.constgen)
    
    
    def output_patterns(self):
        s = ''
        for p in self.patterns:
            s += str(p)
            s += "TID(+):"+str(p.tidl_pos) + "\n"
            s += "TID(-):"+str(p.tidl_neg) + "\n"
        return s
        
    def create_dbs(self, nb=None):
        """create databases with empty sequences
        - nb Number of sequences to create (if None, use self.nbex)
        Both database have the exact same number of sequences
        """
        if not nb is None:
            self.nbex=nb
        
        #generate two sets of nbex empty sequences
        self.db_pos=[]
        self.db_neg=[]
        for i in range(self.nbex):
            self.db_pos.append( sequence(self.l) )
            self.db_neg.append( sequence(self.l) )
    
    def add_pattern(self, p, gr=None, th=None, exclusion_lists=([],[])):
        """add manually a pattern
        - p a chronicle-pattern to add to databases
        - gr specify a growth rate for pattern p (if not, random gr is generated around the default generator gr), could be inf
            * if gr>1: the pattern is more frequent in positives
            * if gr<1: the pattern is more frequent in negatives
        - th specify a frequency threshold for pattern in negatives (if not, the default generator th is used)
            Note that the frequency is not checked in positive examples database !
        - exclusion_lists: a pair of list (pos, neg) of sequences id that must not be used to put patterns
        
        if numbers of occurrences are not compatible with the db sizes, the pattern is not added
        """
        if not isinstance(p, chronicle):
            return
        
        
        #generate a random growth rate for the pattern: very narrow normal distribution
        if gr==None:
            gr = np.random.normal(self.gr,0.02)
        if th==None:
            th=self.th

        #number of positive examples
        if gr<=0:
            nbocc_pos=0
            # number of occurrences generated (randomly above the  threshold)
            #   -> the geometric low is used to randomize the number of occurrences
            #   -> parameters of the law must be >=1.0 (reason why min()
            #   -> 10/nbex: the more you have numbers, the lower is the geometric law parameters (ie high values are more probable)
            nbocc_neg = self.nbex*self.th  + np.random.geometric( min(1.0,10.0/float(self.nbex)) )-1
            nbocc_neg = min(nbocc_neg, self.nbex)
        elif gr==float("inf"):
            nbocc_neg=0
            nbocc_pos = self.nbex*self.th  + np.random.geometric( min(1.0,10.0/float(self.nbex)) )-1 # number of occurrences generated (randomly above the threshold)
            nbocc_pos = min(nbocc_pos, self.nbex)
        else:
            nbocc_neg = self.nbex*self.th  + np.random.geometric( min(1.0,10.0/float(self.nbex)) )-1
            if nbocc_neg>self.nbex:
                warnings.warn("*** #occurrences error (neg): chronicle not inserted ***")
                return
            nbocc_pos = gr*nbocc_neg
            if nbocc_pos>self.nbex:
                warnings.warn("*** #occurrences error (pos): chronicle not inserted ***")
                return
            
        self.patterns.append(p)
        
        ### random attribution of patterns to sequences
        # for positives
        vec = range(0,self.nbex)    # get a set of id
        for i in exclusion_lists[0]:# remove sequences that must not be used
            vec.remove(i)
        np.random.shuffle( vec )    # shuffle it a little bit
        patpos = vec[:int(nbocc_pos)]   # take just the required number of sequenced (at the beginning)
        p.tidl_pos = patpos
        for pos in patpos:
            try:
                self.db_pos[pos].add_pattern( p )
            except IndexError:
                warnings.warn("*** index error: "+str(pos)+" ***")
                pass
                
        # for negatives
        vec = range(0,self.nbex)    # get a set of id
        for i in exclusion_lists[1]:# remove sequences that must not be used
            vec.remove(i)
        np.random.shuffle( vec )    # shuffle it a little bit
        patpos = vec[:int(nbocc_neg)]   # take just the required number of sequenced (at the beginning)
        p.tidl_neg = patpos
        for pos in patpos:
            try:
                self.db_neg[pos].add_pattern( p )
            except IndexError:
                warnings.warn("*** index error: "+str(pos)+" ***")

    def generate(self, l=None):
        """Generation of sequences
        - l mean length of sequences
        """
        #update the parameters of the generation
        if not l is None:
            self.l=l
            #changed the required length for sequences
            for seq in self.db_pos:
                seq.requiredlen=l
            for seq in self.db_neg:
                seq.requiredlen=l
                
        #generate sequences
        for seq in self.db_pos:
            seq.self_generate(self.itemgen)
        for seq in self.db_neg:
            seq.self_generate(self.itemgen)
        

        

if __name__ == "__main__":
    
        ##Generate two databases with discriminant sequences
        generator=discr_db_generator(maxitems)
        generator.create_dbs(seqnb)
        
        for i in range(nbpatterns):
            #generate a pattern pat
            pat = generator.chrogen.generate(lengthpatterns,cd)
            #add pat
            generator.add_pattern( pat, growthrate, threshold )
            
            #create a modified version of pat
            pat2 = generator.chrogen.generate_similar(pat)
            #add pat2 with the reverse growthrate, and exclude adding the "opposite pattern" in same sequences
            generator.add_pattern( pat2, 1.0/float(growthrate), threshold, (pat.tidl_pos, pat.tidl_neg) )
        
        #generate the sequences
        generator.generate(length)
        
        ##write output
        
        filename=outputfile.rsplit(".",1)
        fout = open(filename[0]+"_pos.dat", "w")
        for s in generator.db_pos:
            fout.write(str(s))
        fout.close()
        fout = open(filename[0]+"_neg.dat", "w")
        for s in generator.db_neg:
            fout.write(str(s))
        fout.close()
        
        #output patterns
        filename=outputfile.rsplit(".",1)
        outputfile=filename[0]+".pat"
        fout = open(outputfile, "w")
        fout.write(generator.output_patterns())
        fout.close()

    
