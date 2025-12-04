#ifndef PREFIXTREE_H
#define PREFIXTREE_H
#include "eventseq.h"
#include "treeeventnode.h"
#include "treeeventmap.h"
#include<R.h>
#include "eventdictionary.h"
#include "constraint.h"

class TreeEventNode;


/**
 Prefix Tree base node
*/


class PrefixTree {
    //Start of the tree
    TreeEventMap child;
public:
    //Ctor
    PrefixTree();
    //Dtor
    virtual ~PrefixTree();
   // void addSequence(Sequence *s,const double &maxGap,const double &windowSize, const double & ageMin, const double & ageMax,const double & ageMaxEnd, const int& k);
    void addSequence(Sequence *s, Constraint *cst, const int& k);
    void simplifyTree(double minSup);
    int countSubsequence(double minSup);
    //Give an overview of this tree (paramètre prof==profondeur, interne)
    void print();
    //Type of this event
    void getSubsequences(SEXP result,double * support, int *index, SEXP classname, EventDictionary * ed);
    void clearSupport() {
        this->child.clearSupport();
    }
};

#endif // PREFIXTREE_H
