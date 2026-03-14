# Random sequences generation

Random sequence generators for sequential pattern mining algorithm evaluations.

The principle of this generators are the following:
* generate a collection of random patterns
* assign randomly the pattern to the transaction id (according to a minimal frequency threshold)
* generate sequence satisfying the contraints (patterns, length)

The module includes several generators depending of the desired characteristics of the data and patterns:
* sequences generation and temporal sequence generation: in the second case, timestamps are generated. The large majority of our simulators generates sequences of items (and not of itemsets).
* hidden patterns could have different shapes: sequential patterns, temporal sequential patterns, chronicles: 
* different databases characteristics: random databases, databases containing with frequent patterns, database with negated patterns (ensure that patterns does not occurs)


### Sequential patterns

A sequential pattern is an order set of items, for instance `3>7>5`.
Generating a sequence containing this pattern ensures that it holds the events `3`, `7` and `5` in this order. It is possible to add a maxgap constraint that will avoid to have more than `mg` items between two occurrences of the 

The generator of sequential pattern randomly choose the pattern according to the random item generator.

### Chronicle patterns

A chronicle is a graph of items (with possible repetition) linked by temporal constraints. 

The generation of the random chronicle is set up by the following important parameters
* the generation of items (how many and how)
* the density of temporal constraints (default 0.3): it indicates the mean proportion of constraints to draw
* the characteristics of temporal constraints which are drawn with two Gaussian distributions for start time and duration.

### Temporal patterns

Temporal patterns are derived from the chronicles. We generate random chronicles such that there is temporal constraints only between successive items and that this constraints are only positive (sequential).

### Negative patterns

Negative patterns are generated randomly in two phases
1. the classical temporal patterns are generated
2. the dataset is analyzed to add at most one negative item to a pattern (the position in the pattern is randomly chosen, and the item it randomly chosen within admissible negative items)

In the second phase, the admissible negative items between position p_1 and p_2 are those that are not occurring at least `mlambda` times and at most `Mlambda` times between occurrences of p_1 and p_2 items.
The `mlambda` will refer to the frequency of the negative patterns, while `Mlambda`  will refer to the frequency of the positive partner (see our work on negative patterns for more details).

## Main command for random sequence generator

The main program that can be used to generate random sequences is `generation.py`.

The program has no input file.

The default output files are the following
* the sequence dataset (`output.dat`): one sequence per line, each event of the sequence is a pair `(X,T)` where `T` is a timestamps and `X` is the item. The example below illustrates a dataset of 3 sequences
```
(18, 0) (10,1) (4,2) (6,3) (17,4) (9,5)
(8, 0) (18,1) (9,2) (17,3) (9,4) (5,5) (13,6) (12,7)
(10, 0) (9,1) (4,2) (13,3) (17,4)
```
* the hidden patterns (`output.pat`): describe the randomly generated patterns and give in which pattern it occurs.

The following example illustrates the output while sequential patterns are used (patterns are simply sequence of items). For instance, `P1` is a pattern `[5, 19, 12]` that occurs in sequences 0 and 1 of the generated dataset.
```
P0, [6, 0, 13, 8]: [0]
P1, [5, 19, 12]: [1, 0]
P2, [10, 4, 19, 0]: [2, 0]
```
The following example illustrates the case of the use of chronicles. It details the set of items and the generated temporal constraints between two items (identified by their position in the item list.
```
C0: {[10, 14, 3]}
0,1: (19.466661898474179, 157.86859909433014)
0,2: (113.36645558944412, 288.67537344267947)
1,2: (93.899793690969943, 130.80677434834934)
tid:[0, 1]
```

### Requirements/Installation

* Python3
* numpy
* scipy

No smart pip installation, etc ...

### Examples to generate database of sequences

* Sequential patterns
```
generation.py -t 'sequential' -n 1000 -l 30 --np=6 --mg=3
```
This command generates 1000 sequences of mean length 30 with 6 hidden sequential patterns, and a maxgap constraint of 3

* Temporal patterns

```
generation.py -t 'temporal' -n 1 -l 300 --np=6
```
This command generates 1 sequence of length 300 with 6 hidden patterns (at most).

### Full list of command parameters

```
generation.py -n <val> -l <val> --np=<val> --lp=<val> --th=<val> -d <val> -o <outputfile> --asp
    * t: type of the pattern hidden in sequences 'sequential', 'temporal', 'negative', 'chronicle' or 'random' (default sequential)
    * n: number of sequences (default: 100)
    * l: default length of the sequence (default: 10)
    * d: dictionary size (number of different items) (default: 20)
    * mg: maxgap constraint: maximum number of items between successive items of the pattern within the sequence (default None, without option --r)
    * o output filename
    * np: number of patterns (default 1),
    * lp: mean length of patterns (default 5),
    * c: constraint density (mean number of constraints) (default: 0.3)
    * th: threshold of patterns, minimum occurrence frequency in the database (default 0.1),
    * --asp: activate ASP output
    * --mlambda / --Mlambda: minimum and maximum lambda thresholds (only for negative patterns). Lambda is the proportion of sequences that does not have its negative item, among sequences that support the positive pattern
    
    The generate generate random chronicle patterns and hide them in sequences of the database. Note that the given list of patterns may not be complete with regard to the generated dataset: generation of random sequences/items may create additional occurrences !!
    
Usage example:
    - Generation of datasets for frequent patterns mining
    $ python generation.py -l 10 -d 20 -c 0.6 -o output.dat
    This command generates two files:
        - output.dat: IBM format or a space separated file (one transaction per line)
        - output.pat: description of the hidden patterns (for each hidden pattern, you have its description, a sequence of events, and its locations in the database, ie the ids of the transaction in which it is!)
        
    - Generation of dataset for discriminant pattern mining
    $ python generation.py -l 10 -d 20 -c 0.6 --discr -g 3 -o output.dat
    This command generates three files:
        - output_pos.dat: database of sequences for positive class (IBM format or 1 line per sequence)
        - output_neg.dat: database of sequences for negative class 
        - output.pat: description of the hidden patterns
        
    - Generation of dataset for negative pattern mining
    $ python generation.py -l 10 -d 20 --neg -o output.dat
    This command generates two files:
        - output.dat: database of sequences with 
        - output.pat: description of the hidden patterns (linear chronicles, a single negative event)
```

## Implement your own random sequence generator

It is possible to implement your own sequence generator by inherent the main classes of the `db_generator` module:
* a new class of pattern describing what are the shape of the pattern to hide
* a new class of sequence describing how to generate a sequence that holds your own pattern class
* two simple factories (sequence/pattern) that are used by the database generator

## Author(s)

Thomas Guyet -- AGROCAMPUS-OUEST/IRISA

## License

This project is licensed under the LGPL licence -- see the LICENSE.md file for details


