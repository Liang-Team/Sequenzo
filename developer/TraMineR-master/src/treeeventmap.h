#ifndef TREEEVENTMAP_H
#define TREEEVENTMAP_H
//#include "treeeventnode.h"
//#include <map>
#include "eventseq.h"
extern "C"{
	#include <R.h>
	#include <Rinternals.h>
}

//using namespace std;

//Forward declaration
class TreeEventNode;
//Iterateur pour la map
typedef std::map<int,TreeEventNode*>::iterator TreeEventMapIterator;
//Définition de type pour aleger le code
//Une map sur le type d'événements, et la classe événement
class TreeEventMap: public std::map<int,TreeEventNode*> {

public:
TreeEventMap():std::map<int,TreeEventNode*>() {}
    ~TreeEventMap() {}
    void simplifyTreeMap(const double &minSup);
    void print(const int & prof, const bool& isbrother);
    int countSubsequence(double minSup);
    void getSubsequences(SEXP result,double * support, Sequence *s2, int *index,const double &step, SEXP classname,EventDictionary * ed);
    void clearAllPointers();
    void clearSupport();

};

#endif // TREEEVENTMAP_H
