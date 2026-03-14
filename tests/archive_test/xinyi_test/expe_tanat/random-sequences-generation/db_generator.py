#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Module containing abstract classes for random structured sequence generator.

A sequence is basically an ordered set of timestamp items.

Both patterns and sequences are randomly generated. A sequence can holds several occurrences of different pattern (but not of the same pattern).

The principle of the generator is the following:
- generate a collection of random patterns
- assign randomly the pattern to the transaction id (according to a minimal frequency threshold)
- generate sequence satisfying the contraints
    - pattern that it must contain (including gap constraints)
    - mean length

The generator can ensure that the patterns occurs in some sequences with a given minimum number of occurrences.

To derive another simulator, your must create classes that inherit from the class of this module (see seqdb_generator.py for a simple example):
* The inherited pattern class with the specificities of your patterns
* The inherited sequence class must implement the self_generation() process
* Both classes must have consistant factories (sequence_generator and pattern_generators)
* The inherited db_generator must be initialized with your own parameters and classes
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

class item_generator:
    """Classe to randomly generate items
    
    There two possibilities to generate items: the uniform distribution of the items or a gaussian distribution. The gaussian distribution enables to assume that some items are more rare or more frequent than the others (that seems to be more realistic in most cases)
    """
    def __init__(self, n=100, fl="gaussian"):
        self.flaw=[]    # distribution for the item frequency: "uniform" or "gaussian"
        self.nb=n       # number of items
        if fl=="uniform":
            self.flaw=np.ones(self.nb)/self.nb
        elif fl=="gaussian":
            vec = list(range(0,self.nb))
            vec = [v/float(self.nb) for v in vec]
            sigma=.05
            mu=0.5
            for v in vec:
                self.flaw.append(np.exp( - (v-mu)*(v-mu)/(2*sigma*sigma)) )
            np.random.shuffle(self.flaw)
            self.flaw=self.flaw/sum(self.flaw) #normalisation
        else:
            warnings.warn("*** Unknown distribution law ***")
        
    
    def generate(self):
        """ Generates a random item according to the item distribution modeling
        """
        return np.random.choice(self.nb, 1, p=self.flaw)[0]

class pattern:
    """Sequential pattern (not temporal)
    """
    npat=0 #total number of patterns (used to generate pattern id)
    
    def __init__(self):
        self.sequence = []      # description of the pattern
        self.tidl=[]            # list of the transactions in which the pattern occurs
        self.maxgap = None      # no constraint if None, 
        self.pid=pattern.npat   # pattern id
        pattern.npat += 1


    def append(self, item):    
        """ Add an itemset at the end of the pattern
        
        :param item: item to append to the pattern
        """
        self.sequence.append(item)
    
    def occurs(self, tid):
        """Add a transaction id in which the pattern is actually occurring
        """
        self.tidl.append(tid)
    
    
    def __len__(self):
        """ Length of the patterns (number of items)
        """
        return len(self.sequence)
        
    def __str__(self):
        return "P"+str(self.pid)+", "+str(self.sequence) + ": " + str(self.tidl)


class pattern_generator:
    """Sequential pattern factory
    
    See also
    --------
    pattern
    """
    
    lpat=4
    
    def __init__(self, itemgenerator, mg=None):
        self.item_gen = itemgenerator
        self.mg=mg
    
    def generate(self, l=5):
        """Random generation of one sequential pattern of length l
        
        :param l: length of the pattern
        :return: a pattern
        """
        pat = pattern()
        for i in range(l):
            item = self.item_gen.generate()
            pat.append(item)
        pat.maxgap=self.mg
        return pat

class sequence:
    """Abstract representation of a sequence that can self generate.
    """
    nbs=0
    
    def __init__(self, rl =20, d=1000):
        self.patterns = []  #this is a list to be indexed (instead of a set)
        self.duration = d       # duration of the sequence
        self.seq=[]         #list of couple (t,e) where t is a timestamp and e an item
        self.requiredlen=rl
        self.id=sequence.nbs
        sequence.nbs+=1

    def add_pattern(self, p):
        """Add a pattern without redundancy
        
        :param p: pattern to be included in the sequence
        """
        if not p in self.patterns:
            self.patterns.append(p)

    def __patcontentlen__(self):
        """ Compute the sum of pattern length 
        """
        total=0
        for p in self.patterns:
            total += len(p)
        return total
    
       
    def self_generate(self, itemgen=None):
        """ Generate the sequence of items from the patterns it contains and according to the require length
        
        **Method to implement in inherited classes**
        
        :param itemgen: random itemset generator used to mix the pattern occurrences with random items
        """ 
        if not itemgen is None:
            itemgen = self.item_gen
        
        return
        
    def write_IBM(self):
        """sequence format: IBM format
        """
        if len(self.seq)==0:
            return ""
        s=""
        for item in self.seq:
            s+= str(self.id)+" " + str(item[0]) + " " + str(1) + " " + str(item[1]) + "\n"
        return s
        
    def write_line(self):
        """Write out function: 1 sequence in a line
        """
        if len(self.seq)==0:
            return ""
        s = "("+str(self.seq[0][0])+","+str(self.seq[0][1])+")"
        for c in self.seq[1:]:
            s += " ("+str(c[0])+","+str(c[1])+")"
        return s
        
    def write_pureseq(self):
        """string representation of a sequence without timestamps
        """
        s=str(self.seq[0][0])
        for e in self.seq[1:]:
            s += ","+str(e[0])
        return s
        
    def write_tanatcsv(self):
        """Write out function: csv format with 1 event per line.
        Format: id, stime, etime, event. Each (item, pos) in seq -> stime=pos, etime=pos+1, event=item.
        """
        if len(self.seq)==0:
            return ""
        s = ""
        for c in self.seq:
            item, pos = c[0], c[1]
            s += f"{self.id},{pos},{pos+1},{item}\n"
        return s
        

    def write_asp(self):
        """ASP string representing the sequence.
               
        This function can be used as an alternative 
        The function uses predicate seq/3, where seq(S,P,I) means "sequence S at position P has the item I"
        :return: return an ASP string
        """
        so=''
        i=0
        for c in self.seq:
            so += "seq("+str(self.id)+","+str(c[0])+","+str(c[1])+').'
            i+=1
        return so
    
    def __str__(self):
        """ String cast
        """
        return self.write_line()
        #return self.write_IBM()
        #return self.write_pureseq()


class sequence_generator:
    """Sequence generator
    
    See also
    --------
    sequence
    """
    
    def __init__(self):
        pass
    
    def generate(self,l):
        """Generate an empty sequences of length l
        """
        return sequence(rl=l)


class db_generator:
    """Database generator
    """

    def __init__(self, item_generator, sequence_generator, pattern_generator):
        """
        
        """
        self.patterns = []      # set of patterns
        self.db = []            # database of sequences
        self.stats={}           # collected statistics
        
        self.nbex = 1000        # number of sequences in the database
        self.nbpat = 5          # number of the patterns
        self.th = 0.20          # pattern threshold
        
        self.item_generator=item_generator
        self.sequence_generator=sequence_generator
        self.pattern_generator=pattern_generator
    
    def generate_patterns(self):
        """ The function generates a set of self.nbpat patterns of mean length self.lpat.
        It uses the self.patgen pattern generator
        - Requires the self.patgen to be defined (return an empty list if not)
        """
        if self.pattern_generator == None:
            warnings.warn("*** undefined chronicle pattern generator ***")
            return []
        patlen = [int(np.floor(v)) for v in np.random.normal(pattern_generator.lpat, max(0.5,float(pattern_generator.lpat)/10.0), self.nbpat)]
        patterns = [self.pattern_generator.generate(l) for l in patlen]
        return patterns
        
    def generate_sequences(self):
        self.db=[]
        sequence.nbs=0
        for i in range(self.nbex):
            self.db.append( self.sequence_generator.generate(self.l) )
    
    def output_patterns(self):
        """Print out patterns function
        
        Return a string with the patterns and their tid lists
        """
        s = ''
        for p in self.patterns:
            s += str(p)+"\n"
        return s
    
    def generate(self, nb=None, l=None, npat=None, th=None):
        """Generation of the sequence database
        
        :param nb: number of sequences
        :param l: mean length of the sequence
        :param npat: number of patterns
        :param th: frequency threshold (number of sequences covering each pattern)
        """
        #update the parameters of the generation
        if not nb is None:
            self.nbex=nb
        if not l is None:
            self.l=l
        if not th is None:
            self.th=th
        if not npat is None:
            self.nbpat=npat
            
        #collect "real" statistics about the generated sequences
        self.stats={}
        
        #generate a set of nbex empty sequences
        self.generate_sequences()
        self.stats["nbex"]=self.nbex
            
        #generate self.nbpat random patterns using the generator
        self.patterns=self.generate_patterns()
        self.stats["nbpat"] = len(self.patterns)
        
        nbM=-1
        nbMean=0
        nbm=self.nbex
        #attribute transactions to the patterns
        for p in self.patterns:
            nbocc = self.nbex*self.th  + (np.random.geometric(0.15)-1) # number of occurrences generated (randomly above the threshold)
            nbocc = min(nbocc, self.nbex)
            #generation of a random set of sequences of size nbex (ensure no repetitions)
            vec = list(range(0,self.nbex))    # get a set of id
            np.random.shuffle( vec )    # shuffle it a little bit
            patpos = vec[:int(nbocc)]   # take just take the require number of sequence (at the beginning)
            p.tidl = patpos
            
            nb=0
            for pos in patpos:
                try:
                    self.db[pos].add_pattern( p )
                    nb+=1
                except IndexError:
                    warnings.warn("*** index error: "+str(pos)+" ***")
                    pass
            nbM=max(nbM,nb)
            nbM=min(nbm,nb)
            nbMean+=nb
        if self.stats["nbpat"]!=0: 
            nbMean/=self.stats["nbpat"]
        else:
            nbMean=0
        self.stats["nboccpat_max"]=nbM
        self.stats["nboccpat_min"]=nbm
        self.stats["nboccpat_mean"]=nbMean
        
        #generate sequences
        for i in range(self.nbex):
            self.db[i].self_generate(self.item_generator)
        
        return self.db

