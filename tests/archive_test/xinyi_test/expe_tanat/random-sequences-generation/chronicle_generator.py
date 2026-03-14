#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Random sequence generator based on chronicle models.

The program generates two types of datasets:
- dataset containing frequent patterns
- positive and negative sequences

Moreover, it is possible to generate dataset of patterns with negations.
    -> in this case, chronicles are simply positive, linear patterns !
    The principle of the generation of a negative patterns is first to generate a dataset containing 
    frequent chronicles randomly generated (with linear, positive temporal constraints only). Each 
    generated chronicle is complemented with the choice of a negative position
    While the database has been generated, the generated dataset is analyzed to identify which event can
    be choosen to be negative at the negative position.
    This event has to be frequent enough (in order to be detected by e-NSP), but not to much, such that the 
    negative pattern is frequent! This is set up by the two thresholds Mlambda and mlambda!
"""

__author__ = "Thomas Guyet"
__copyright__ = "Copyright 2019, AGROCAMPUS-OUEST/IRISA"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "Thomas Guyet"
__email__ = "thomas.guyet@irisa.fr"

import warnings
import numpy as np
import sys, getopt
import warnings
import scipy.sparse.csgraph

from db_generator import *

class chronicle(pattern):
    """Class for a chronicle pattern modeling
    
    The sequence attribute represents a sorted list of items (e)
    """
    
    def __init__(self):
        super().__init__()   
        self.negative_item=None
        self.negative_position=None
        self.occurred_possible_neg_items={} # { item : [seqid, ...] }
        
        self.tconst={}  #temporal constraints,
                        # keys: couple (ei,ej) where ei is a index in the item
                        #   in the multiset
                        # values; couple (mu,var) where mu is the 
        self.inconsistent = False
        self.tidl_neg=[]    # TID list in negative dataset

        
    def add_possible_neg_item(self, item, seqid):
        self.occurred_possible_neg_items.setdefault(item,set())
        self.occurred_possible_neg_items[item].add(seqid)
        
        
    def add_item(self, item):
        """Add an item to the chronicle and return the id of the added event
        The function creates all infinite constraints, without variability
        - the id of the event correspond to the order of added items
        """
        self.sequence.append(item)
        id = len(self.sequence)-1
        for i in range(len(self.sequence)):
            self.tconst[(i,id)]= (-float("inf"),float("inf"))
        return id
        
    def add_constraint(self, ei, ej, constr):
        """Add a constraint-template to the chronicle pattern
        - ei, ej: index of the events in the multiset
        - constr: a 4-tuple (mu-start, var-start, mu-end, var-end) of the mean and variance of temporal constraint
        """
        if not type(constr) is tuple:
            print ("error: constraint must be a tuple (=> constraint not added)")
            return
            
        if len(constr)!=2:
            print ("error: constraint must have 4 values (=> constraint not added)")
            return
            
        try:
            self.tconst[(ei,ej)] = constr
        except IndexError:
            print ("add_constraint: index_error (=> constraint not added)")
        
    def __getitem__(self, i):
        """return the item at position i in the multiset if i is an integer
        and return the constraint between i[0] and i[1] if i is a couple
        """
        if not type(i) is tuple:
            return self.sequence[i]
        else:
            return self.tconst[(min(i[0],i[1]),max(i[0],i[1]))]
        
    def __len__(self):
        return len(self.sequence)
            
    def __str__(self):
        s = "C"+str(self.pid)+": {"+str(self.sequence) + "}\n"
        for i in range(len(self.sequence)):
            for j in range(i+1,len(self.sequence)):
                s += str(i) + "," + str(j) + ": " + str(self.tconst[(i,j)])+"\n"
        if not self.negative_item is None:
            s+= "neg item " + str(self.negative_item)+ " after event at position " + str(self.negative_position)+"\n"
        elif not self.negative_position is None:
            s+= "neg pos: " + str(self.negative_position)+"\n"
        self.tidl.sort()
        s+="tid:"+str(self.tidl)+"\n"
        if len(self.tidl_neg)>0:
            s+="tid neg:"+str(self.tidl_neg)
        return s

    def minimize(self):
        #construction of distance graph
        mat=np.matrix( np.zeros( (len(self),len(self)) ))
        for i in range(len(self)):
            for j in range(i+1,len(self)):
                mat[i,j] = self.tconst[ (i,j) ][1]
                mat[j,i] = -self.tconst[ (i,j) ][0]
        try:
            matfw = scipy.sparse.csgraph.floyd_warshall( mat )
            #construction of simpllified chronicle
            for i in range(len(self)):
                for j in range(i+1,len(self)):
                    self.tconst[ (i,j) ] = (- matfw[j,i], matfw[i,j])
        except:
            warnings.warn("*** Minimisation: Inconsistent chronicle ***")
            self.inconsistent = True


class constraint_generator:
    """Chronicle constraint generator
    
    It randomly generates temporals constraints for chronicles, ie temporal intervals.
    The interval boundaries are uniformly distributed within the limits. 
    """
    def __init__(self, minstart=-100, maxstart=100, minduration=0.1, maxduration=200):
        self.ms=minstart
        self.Ms=maxstart
        self.md=minduration
        self.Md=maxduration
    
    def generate(self, ctype=""):
        if ctype=="after":
            s= np.random.uniform(0, self.Ms)
            f= s + np.random.uniform(self.md, self.Md)
        else:
            s= np.random.uniform(self.ms, self.Ms)
            f= s + np.random.uniform(self.md, self.Md)
        c=(s, f)
        return c


class chronicle_generator:
    """Factory class for chronicles
    
    It provides a function to generates chronicles with consistant, minimal constraints; and a function to generate disturbed chronicles
    In this class negation are not used
    """
    
    maxtest=10
    
    def __init__(self, ig, cg, cd=0.3):
        self.itemGenerator = ig
        self.constraintGenerator = cg
        self.constraintdensity = cd
        
    def __raw_generate__(self, l):
        chro = chronicle()
        for i in range(l):
            item = self.itemGenerator.generate()
            chro.add_item(item)
        for i in range(l):
            for j in range(i+1,l):
                if np.random.rand()<self.constraintdensity:
                    c=self.constraintGenerator.generate()
                    chro.add_constraint(i, j, c)
        return chro
    
    def generate(self,l,cd=None):
        """
        Function that generate a random consistent, minimal chronicle.
        Generate and test approach: generate a random chronicle, test if it is consistent (at most maxtime times, otherwise None is returned)
        :param l: mean size of the multiset a chronicle pattern
        :param cd: constraint density
        """
        if not cd is None:
            self.constraintdensity = cd
        T=0
        while T<chronicle_generator.maxtest:
            chro = self.__raw_generate__(l)
            #consistency checking and minimisation
            chro.minimize()
            if not chro.inconsistent:
                return chro
            T+=1
        raise NameError("Impossible to generate a consistent chronicles")
        return None
        
        
    def generate_similar(self, C, proba=[0.1,0.8,0.1]):
        """function that generates a chronicle similar to C
        
        representing the proba of modifications
            1- removing items (defailt 0.1)
            2- modyfying constraints (default 0.8)
            3- adding item (default 0.1)
        
        what can change in the generated chronicle:
            - temporal bounds
            - multi-set (add or remove items)
        
        :param C: is a chronicle
        :param proba: list of 3 values representing the proba of modifications
        :return: a chronicle
        """
        if not isinstance(C, chronicle):
            return
            
        chro = chronicle()
        
        ########## RANDOM MODIFICATION SELECTION  ###############
        removeitem=False
        modify_tconst = False
        additem=False
        
        #proba normalisation
        vec=np.array(proba)
        proba = vec/np.sum(vec)
        alea=np.random.rand()
        i=0
        while i<3 and alea>proba[i]:
            alea -= proba[i]
            i+=1
        if i==0:
            removeitem=True
        elif i==1:
            modify_tconst = True
        else:
            additem=True
        
        ################ CHRONICLE MODIFICATIONS #############
        l=len(C.multiset)
        if removeitem:
            idr = np.random.randint( l )
            for i in range(l):
                if i==idr:
                    continue
                chro.multiset.append( C[i] )
            #copy constraints (removing idr + decay references)
            for i in range(idr):
                for j in range(i+1,idr):
                    chro.add_constraint(i, j, C[(i,j)] )
                for j in range(idr+1,l):
                    chro.add_constraint(i, j-1, C[(i,j)] )
            for i in range(idr+1,l):
                for j in range(i+1,l):
                    chro.add_constraint(i-1, j-1, C[(i,j)] )
                    
        if additem: #add a new item to
            chro.multiset = list(C.multiset)
            chro.tconst = C.tconst.copy()
            ni = self.itemGenerator.generate()
            chro.multiset.append(ni)
            nl = len(chro.multiset)-1
            for j in range( nl ):
                if np.random.uniform(0,1)<self.constraintdensity:
                    c=self.constraintGenerator.generate()
                else:
                    c=(-float("inf"), float("inf"))
                chro.add_constraint(j, nl, c)
        
        if modify_tconst:
            chro.multiset = list(C.multiset)
            chro.tconst = dict(C.tconst)
            j = np.random.randint( 1, l )
            i = np.random.randint( j )
            
            #generate a new random constraint
            c = self.constraintGenerator.generate()
            chro.add_constraint(i, j, c )
        
        ################ CHRONICLE MINIMISATION #############
        chro.minimize()
        return chro
        
        
class negative_chronicle_generator:
    """Factory class that generates a linear chronicle, including one negative item
    
    """
    
    maxtest=10
    
    def __init__(self, ig, cg, cd=0.3):
        """
        :param ig: an itemset generator
        :param cg: a constraint generator that **must generate only positive temporal constraints**
        """
        self.itemGenerator = ig
        self.constraintGenerator = cg
        self.constraintdensity = cd
        if cg.ms<0:
            warnings.warn( "minstart value for the constraintGenerator will not be used\n" )
        
    def generate(self, l, dc=None):
        """ Function that generates a linear chronicle
        
        The generate chronicle will be automatically consistent consitering the positive linear constraints we impose (on the contrary to the other chronicle generator)
        :param l: mean size of the multiset a chronicle pattern
        :param dc: unused
        """
        chro = chronicle()
        for i in range(l):
            item = self.itemGenerator.generate()
            chro.add_item(item)
        for i in range(l):
            c=self.constraintGenerator.generate(ctype="after")
            chro.add_constraint(i, i+1, c)
            
        chro.minimize() #compute the complete graph of temporal constraints
        assert(not chro.inconsistent)
        
        #generate a (unique) position in which to place a negative item
        if l>1:
            chro.negative_position = np.random.randint(0,l-1)
        
        return chro
        


class chro_sequence(sequence):
    """ A sequence is a list of time-stamped items
    
    All sequences start at 0 and there duration is defined while constructed (self.duration)
    """
    gen_int_timestamped=True #if True, timestamped will be integers
    
    def __init__(self, rl =20, d=1000):
        super().__init__(rl,d)
    
    def interval_substraction(self,intervals, int_rest):
        """
        :param intervals: list of intervals
        :param int_rest: interval to substract to each element of intervals
        :return: list of modified intervals
        """
        new_inter = intervals[:]
        for inter in intervals:
            if inter[0]<int_rest[0]:
                if inter[1] > int_rest[1]:
                    new_inter.append( (inter[0], int_rest[0]) )
                    new_inter.append( (int_rest[1], inter[1]) )
                elif inter[0] > int_rest[0]:
                    new_inter.append( (inter[0], int_rest[0]) )
                else:
                    new_inter.append( (inter[0], inter[1]) )
            elif inter[0]<int_rest[1]:
                if inter[1] > int_rest[1]:
                    new_inter.append( (int_rest[1], int_rest[0]) )
            else:
                new_inter.append( (inter[0], inter[1]) )
        return new_inter
        
        
    def self_generate(self, item_gen):
        """ Generate the sequence of items from the patterns it contains and according to the required length
    
        Timestamped are generated according to a random uniform law within the duration of the sequence
        The function enable to generate negative pattern
    
        :param item_gen: random item generator to mix the pattern occurrences with random items.
        """
    
        # no patterns inside: fully random sequences
        if len(self.patterns)==0:
            l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
            for i in range(l):
                #random item (according to an item generator)
                item = item_gen.generate()
                #random timestamp
                timestamp = np.random.uniform(self.duration)
                self.seq.append( (timestamp,item) )
                if chro_sequence.gen_int_timestamped:
                    self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
                self.seq.sort(key=lambda tup: tup[0])
            return
            
        negative_period={}
        
        totcreated=0
        for p in self.patterns:
            occurrence=[] #timestamped of the occurrence
            t=np.random.uniform(0,self.duration/2) #chronicle start at the beginning of the sequence
            occurrence.append(t)
            self.seq.append( (t,p[0]) )
            
            for i in range(1,len(p)):
                # construct a interval in which i-th event could occurs
                interval=[0,self.duration]
                for j in range(i):
                    lc = p[ (j,i) ] #chronicle constraint between j and i
                    #interval intersection (constraints conjunction)
                    if interval[0]<occurrence[j]+lc[0]:
                        interval[0]=occurrence[j]+lc[0]
                    if interval[1]>occurrence[j]+lc[1]:
                        interval[1]=occurrence[j]+lc[1]
                    
                #generate a timestamp in the interval
                if interval[0]>=interval[1]:
                    warnings.warn("*** chronicle is not consistent ***")
                    self.seq=[]
                    return
                
                t = np.random.uniform(interval[0],interval[1])
                self.seq.append( (t,p[i]) )
                occurrence.append(t) #timestamp of the i-th item in the chronicle occurrence
                
            if not p.negative_position is None:
                if p.negative_position == (len(occurrence)-1):
                    negative_period[p]=(occurrence[p.negative_position],float("inf"))
                else:
                    negative_period[p]=(occurrence[p.negative_position],occurrence[p.negative_position+1])
                
            totcreated += len(p)
            
        l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
        while totcreated<l:
            #random item (according to an item generator)
            item = item_gen.generate()
            #random timestamp
            timestamp = np.random.uniform(self.duration)
            self.seq.append( (timestamp,item) )
            totcreated+=1
            
            
        #sort the list according to timestamps
        if chro_sequence.gen_int_timestamped:
            self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
        self.seq.sort(key=lambda tup: tup[0])
        
        
        for p in self.patterns:
            if not p.negative_position is None:
                for item in self.seq:
                    if item[0]>negative_period[p][0] and item[0]<negative_period[p][1]:
                        p.add_possible_neg_item(item[1], self.id)
        

class chrosequence_generator(sequence_generator):
    """Factory for sequence based on chronicles
    """
    def __init__(self,d=None):
        self.duration = d
        super().__init__()
    
    def generate(self,l,d=None):
        if d is None and self.duration is None:
            return chro_sequence(l)
        else:
            d = self.duration if d is None else d
            return chro_sequence(l, d)


class chrodb_generator(db_generator):
    """Database generator
    """

    def __init__(self, nbitems=100, l = 30, lp=4, fl="uniform", dc= 0.30, minstart=-100, maxstart=100, minduration=0.1, maxduration=200, seq_duration=1000):
        """Constructor of the db generator
        
        :param nbitems: vocabulary size (default 100)
        :param l: mean length of the sequences
        :param fl: item frequency distribution 'uniform', 'gaussian'
        :param dc: constraint density (if 0: no constraints, if 1 each pair of event are temporaly constraints)
        :param lp: pattern length
        :param minstart, maxstart, minduration, maxduration: temporal constraint characteristics
        :param seqduration: sequence duration
        """
        
        itemgen= item_generator(n=nbitems, fl=fl)
        seqgen = chrosequence_generator(d=seq_duration)
        constrgen = constraint_generator(minstart, maxstart, minduration, maxduration)
        patgen = chronicle_generator(itemgen, constrgen, dc)
        pattern_generator.lpat=lp #change class attribute  
    
        super().__init__(itemgen, seqgen, patgen)
    

class chrodbneg_generator(db_generator):
    """Database generator with negative patterns
    
    The principle of this generation process is first to generate a database with "classical" patterns and then to look for possible adjunction of negative patterns according to the generate sequences.
    With this process, we never fail to generate the database, but may not necessarily generate patterns containing negative items.
    """
    
    def __init__(self, nbitems=100,l = 30, lp=4, fl="uniform", dc = 0.30):
        """Constructor of the db generator
        
        :param n: vocabulary size (default 100)
        :param l: mean length of the sequences
        :param fl: item frequency distribution
        :param dc: constraint density (if 0: no constraints, if 1 each pair of event are temporaly constraints)
        
        """
        self.min_lambda=0.2     # lambda threshold (for negative patterns: maximal proportion of sequence supporting the positive partner without satisfying the negative item)
        self.max_lambda=0.5   
        
        itemgen= item_generator(n=nbitems, fl=fl)
        seqgen = chrosequence_generator()
        constrgen = constraint_generator(0,100,0.1,200)
        patgen = negative_chronicle_generator(itemgen, constrgen, dc)
        pattern_generator.lpat=lp #change class attribute  
    
        super().__init__(itemgen, seqgen, patgen)
    
    def generate(self, nb=None, l=None, npat=None, th=None):
        super().generate(nb, l, npat, th)
        
        #generate negation in patterns
        for p in self.patterns:
            todelete=[]
            for item in p.occurred_possible_neg_items.keys():
                gen_lambda = float(len(p.occurred_possible_neg_items[item]))/float(len(p.tidl)) #proportion of sequences that does not have the item item, among sequences that support the positive pattern
                if gen_lambda < self.min_lambda or gen_lambda > self.max_lambda:
                    todelete.append(item)
                    
            # delete from the possible negative items, those that does not satisfy the lambda constraint
            for item in todelete:
                del( p.occurred_possible_neg_items[item] )

            #random choice of an item that satisfies the lambda thresholds
            if len( p.occurred_possible_neg_items.keys() )>0:
                ritem = np.random.choice( list(p.occurred_possible_neg_items.keys()), 1 )[0]
                p.negative_item = ritem
                #update the list of occurrences
                p.tidl_neg = p.tidl[:]
                p.tidl_neg.sort()
                p.tidl = [item for item in p.tidl if item not in p.occurred_possible_neg_items[ritem] ]
        
        return self.db

if __name__ == "__main__":
    print("====================\nTest Chronicle DB generation\n====================\n")
    generator=chrodb_generator(n=20, lp=3)
    
    sequences = generator.generate(nb=10, l=7, npat=3, th=.2)

    print("======== PATTERNS =======")
    print(generator.output_patterns())
    
    print("======== SEQUENCES =======")
    for s in sequences:
        print(str(s))
        
    print("====================\nTest NEGATIVE Chronicle DB generation\n====================\n")
    generator=chrodbneg_generator(n=20, lp=3)
    
    sequences = generator.generate(nb=10, l=7, npat=3, th=.2)

    print("======== PATTERNS =======")
    print(generator.output_patterns())
    
    print("======== SEQUENCES =======")
    for s in sequences:
        print(str(s))
