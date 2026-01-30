#ifndef TREEEVENTNODE_H
#define TREEEVENTNODE_H
#include "treeeventmap.h"
#include<R.h>
#include <Rinternals.h>
#include "eventseq.h"
#include "constraint.h"


class TreeEventMap;
class TreeEventNode {
    //type of the event
    int type;
    //CurrentSupport
    double support;
    //Last sequence that has incremented the support of this subsequence
    int lastID;
    //Next event at same time (but event bigger!)
    TreeEventMap brother;
    //Next event with gap
    TreeEventMap child;
    static int nodeCount;
public:
    static int getNodeCount();
    //Ctor
    TreeEventNode(const int& t);
    //Dtor
    virtual ~TreeEventNode();
    //Ajoute une séquence et l'ensemble des sous-séquences qui la compose. Méthode récursive (dernier paramètre = paramètre interne)
    //void addSequence(Sequence *s,const double &maxGap,const double &windowSize);
    //Ajoute une séquence et l'ensemble des sous-séquences qui la compose. Méthode récursive (dernier paramètre = paramètre interne)
//    void addSequenceInternal(Sequence *s, SequenceEventNode * en, const double &maxGap,const double &windowSize,const double & ageMax, const double &gapConsumed,  const double& currentAge, const int& k, const int&currentK);
    void addSequenceInternal(Sequence *s, SequenceEventNode * en, Constraint * cst, const double &gapConsumed,  const double& currentAge, const int& k, const int&currentK);

    //Simplifie l'arbre pour enlever l'ensemble des sous-séquences qui ne satisfont pas le support minimum (nb occurrences)
    void simplifyTree(double minSup);
    //Give an overview of this tree (paramètre prof==profondeur, interne)
    void print(const int & prof=0, const bool& isbrother=true);
    //Type of this event
    const int& getType() {
        return this->type;
    }
    //Actual support of this event
    const double& getSupport() {
        return this->support;
    }
    int countSubsequence(double minSup);
    void getSubsequences(SEXP result,double * isupport, Sequence *s, int *index,const double &step, SEXP classname,EventDictionary * ed);
    void clearSupport();
};
#endif // TREEEVENTNODE_H
