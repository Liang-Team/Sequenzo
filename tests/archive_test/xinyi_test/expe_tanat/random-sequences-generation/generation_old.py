#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Random sequence generator main tool: it generates easily and flexibly random sequences based on patterns

::seealso seqdb_generator
"""

__author__ = "Thomas Guyet"
__copyright__ = "Copyright 2019, AGROCAMPUS-OUEST/IRISA"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "Thomas Guyet"
__email__ = "thomas.guyet@irisa.fr"

import seqdb_generator as seqgen
import chronicle_generator as chrogen
import warnings
import sys, getopt
import numpy as np
from random_seq_itemset import gen_seqdb


helpstring="""generation.py -n <val> -l <val> --np=<val> --lp=<val> --th=<val> -d <val> -o <outputfile> --asp
    * t: type of the pattern hidden in sequences 'sequential', 'temporal', 'negative', 'chronicle' or 'random' (default sequential)
    * n: number of sequences (default: 100)
    * l: mean length of the sequence (default: 10)
    * d: dictionnary size (number of different items) (default: 20)
    * mg: maxgap constraint: maximum number of items between successive items of the pattern within the sequence (default None, without option --r)
    * o output filename
    * np: number of patterns (default 1),
    * lp: mean length of patterns (default 5),
    * c: constraint density (mean number of constraints) (default: 0.3)
    * th: threshold of patterns, minimum occurrence frequency in the database (default 0.1),
    * --asp: activate ASP output
    * --pd: activate Pandas csv output
    * --fl: activate fixed length generation (all sequences with fixed length) (for sequential patterns only)
    * --mlambda / --Mlambda: minimum and maximum lambda thresholds (only for negative patterns). Lambda is the proportion of sequences that does not have its negative item, among sequences that support the positive pattern
    
    The generate generate random chornicle patterns and hide them in sequences of the database. Note that the given list of patterns may not be complete with regard to the generated dataset: generation of random sequences/items may create additional occurrences !!
    
Usage example:
    - Generation of datasets for frequent patterns mining
    $ python generation.py -l 10 -d 20 -c 0.6 --ibm -o output.dat
    This command generates two files:
        - output.dat: IBM format or a space separated file (one transaction per line)
        - output.pat: description of the hidden patterns (for each hidden pattern, you have its description, a sequence of events, and its locations in the database, ie the ids of the transaction in which it is!)
        
    - Generation of dataset for discriminant pattern mining
    $ python generation.py -l 10 -d 20 -c 0.6 --discr -g 3 --ibm -o output.dat
    This command generates three files:
        - output_pos.dat: database of sequences for positive class (IBM format or 1 line per sequence)
        - output_neg.dat: database of sequences for negative class 
        - output.pat: description of the hidden patterns
        
    - Generation of dataset for negative pattern mining
    $ python generation.py -l 10 -d 20 --neg --ibm -o output.dat
    This command generates two files:
        - output.dat: database of sequences with 
        - output.pat: description of the hidden patterns (linear chronicles, a single negative event)
"""


def main(argv):
    """Main function
    - parse the command line arguments
    - generate a database in a file
    """
    
    ## Default parameters values
    seqnb=100
    length=10
    maxitems=20
    outputfile="output.dat"
    nbpatterns = 3
    lengthpatterns = 4
    threshold=0.10
    min_lambda=0.20
    max_lambda=0.50
    cd = 0.30
    asp = False
    mg=None
    ibm=False
    tanat=False
    fixedlength = False
    patterntype = "sequential" # ['sequential', 'temporal', 'negative', 'chronicle', 'random']
    
    ## Read parameters
    try:
       opts, args = getopt.getopt(argv,"hn:d:l:c:o:t:i",["pd","ibm","asp","fl","Mlambda=","mlambda=","nbseq=","nbitems=","length=","type=","ofile=", "np=", "lp=", "th=", "mg="])
    except getopt.GetoptError:
       print (helpstring)
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print(helpstring)
           return
       elif opt in ("-t","--type"):
           patterntype = arg
       elif opt in ("-d", "--nbitems"):
           try:
               maxitems = int(arg)
           except:
               warnings.warn("error with argument -d: an integer must be given")
               return
           if maxitems<=0:
               warnings.warn("warning with argument -d: must be strictly positive")
               return
       elif opt in ("-l", "--length"):
           try:
               length = int(arg)
           except:
               warnings.warn("error with argument -l: an integer must be given")
               return
           if length<=0:
               warnings.warn("warning with argument --l: must be strictly positive")
               return
       elif opt in ("-n", "--nbseq"):
           try:
               seqnb = int(arg)
           except:
               warnings.warn("error with argument -n: an integer must be given")
               return
           if seqnb<=0:
               warnings.warn("warning with argument -n: must be strictly positive")
               return
       elif opt in ("-o", "--ofile"):
           outputfile = arg
       elif opt in ("--np"):
           try:
               nbpatterns = int(arg)
           except:
               warnings.warn("error with argument --np: an integer must be given")
               return
       elif opt in ("--lp"):
           try:
               lengthpatterns = int(arg)
           except:
               warnings.warn("error with argument --lp: an integer must be given")
               return
           if lengthpatterns<=0:
               warnings.warn("warning with argument --lp: must be strictly positive")
               return
       elif opt in ("--mg"):
           try:
               mg = int(arg)
           except:
               warnings.warn("error with argument --mg: a positive integer must be given")
               return
           if mg<0:
               warnings.warn("warning with argument --mg: must be positive, ignored")
               mg=None
       elif opt in ("-c", "--constdensity"):
           try:
               cd = float(arg)
           except ValueError:
               warnings.warn("error with argument -c: a float must be given")
               return
           if cd<0 or cd>1:
               warnings.warn("error with argument -c: invalid value, must be in [0,1].")
               return
       elif opt in ("--asp"):
           asp = True
       elif opt in ("--pd"):
           tanat = True
       elif opt in ("-i","--ibm"):
           ibm = True
       elif opt in ("--fl"):
           fixedlength = True
       elif opt in ("--mlambda"):
           try:
               min_lambda = float(arg)
           except:
               warnings.warn("error with argument --mlambda: a float must be given")
               return
           if min_lambda<0 or min_lambda>1:
               warnings.warn("error with argument --mlambda: invalid value, must be in [0,1].")
               return
       elif opt in ("--Mlambda"):
           try:
               max_lambda = float(arg)
           except:
               warnings.warn("error with argument --Mlambda: a float must be given")
               return
           if max_lambda<=0 or max_lambda>1:
               warnings.warn("error with argument --Mlambda: invalid value, must be in [0,1].")
               return
       elif opt in ("--th"):
           try:
               threshold = float(arg)
           except ValueError:
               warnings.warn("error with argument --th: a float must be given")
               return
           if threshold<=0 or threshold>1:
               warnings.warn("error with argument --th: invalid value, must be in ]0,1[.")
               return
       

    if patterntype=='sequential':
        generator = seqgen.seqdb_generator(n=maxitems, mg=mg, lp=lengthpatterns, fixedlength=fixedlength)
    elif patterntype=='temporal':
        generator = chrogen.chrodb_generator(nbitems=maxitems, l = length, lp=lengthpatterns, fl="uniform", dc= cd, minstart=0, maxstart=50, minduration=1, maxduration=20)
    elif patterntype=='temporal_small':
        generator = chrogen.chrodb_generator(nbitems=maxitems, l = length, lp=lengthpatterns, fl="uniform", dc= cd, minstart=0, maxstart=5, minduration=1, maxduration=2, seq_duration=100)
    elif patterntype=='negative':
        generator = chrogen.chrodbneg_generator(nbitems=maxitems, l = length, lp=lengthpatterns, fl="uniform")
        generator.min_lambda = min_lambda
        generator.max_lambda = max_lambda
    elif patterntype=='chronicle':
        generator = chrogen.chrodb_generator(nbitems=maxitems, l = length, lp=lengthpatterns, fl="uniform", dc= cd)
    elif patterntype=='random':
        sequences = gen_seqdb(seqnb, length, maxitems, False)
        
        ## Output sequences
        fout = open(outputfile, "w")
        for s in sequences:
            if ibm:
                fout.write(s.write_IBM())
            elif tanat:
                fout.write(s.write_tanatcsv())
            else:
                fout.write(str(s)+"\n")
        fout.close()
        
        if asp:
            filename=outputfile.rsplit(".",1)
            if len(filename)==1 or filename[1]!="lp":
                outputfile=filename[0]+".lp"
                fout = open(outputfile, "w")
                for s in sequences:
                    fout.write(s.asp())
                    fout.write('\n')
                fout.close()
        return
    else:
        warnings.warn("*** unknown pattern type ***")
        return

    ## Generate database of sequences
    sequences = generator.generate(nb=seqnb, l=length, npat=nbpatterns, th=threshold)
    
    ## Output sequences
    fout = open(outputfile, "w")
    if tanat:
        fout.write("id,stime,etime,event\n")
    for s in sequences:
        if ibm:
            fout.write(s.write_IBM())
        elif tanat:
            fout.write(s.write_tanatcsv())
        else:
            fout.write(str(s)+"\n")
    fout.close()
    
    # Output patterns
    filename=outputfile.rsplit(".",1)
    outputfile=filename[0]+".pat"
    fout = open(outputfile, "w")
    fout.write(generator.output_patterns())
    fout.close()
    
    if asp:
        filename=outputfile.rsplit(".",1)
        if len(filename)==1 or filename[1]!="lp":
            outputfile=filename[0]+".lp"
            fout = open(outputfile, "w")
            for s in sequences:
                fout.write(s.asp())
                fout.write('\n')
            fout.close()


if __name__ == "__main__":
    main(sys.argv[1:])


