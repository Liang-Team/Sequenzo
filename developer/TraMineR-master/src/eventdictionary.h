#ifndef EVENTDICTIONARY_H
#define EVENTDICTIONARY_H
#include <map>
#include <set>
#include <string>
#include <Rinternals.h>
#include "TraMineR.h"



class EventDictionary: public std::map<int,std::string> {
    int numseq;
public:
EventDictionary():numseq(0) {}
    EventDictionary(SEXP flist);
    virtual ~EventDictionary();
    bool codeExists(const int &code) const;
    //int sprint(char * buffer, const char* start, const int&code)const;
    void addSequence() {
        this->numseq++;
    }
    void removeSequence() {
        this->numseq--;
    }
    bool shouldDelete() {
        return this->numseq<1;
    }
    SEXP getDictionary()const;

protected:

private:

};
typedef EventDictionary::iterator EventDictionaryIterator;

class EventSet: public std::set<int>{

public:
	EventSet():std::set<int>(){}
	inline bool contain(const int& code) const{
		return this->find(code)!=this->end();
	}
	void add(SEXP elist);

};
#endif // EVENTDICTIONARY_H
