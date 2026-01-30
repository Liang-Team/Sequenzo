#ifndef LCPDISTANCECALCULATOR_H
#define LCPDISTANCECALCULATOR_H
#include "distancecalculator.h"

class LCPdistance: public DistanceCalculator{
  protected:
	int sign;
  public:
	LCPdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS):DistanceCalculator(normS, Ssequences, seqdim, lenS), sign(0){}
	LCPdistance(LCPdistance *dc):DistanceCalculator(dc), sign(dc->sign){}
	virtual DistanceCalculator* copy(){return new LCPdistance(this);}
	virtual void setParameters(SEXP params){
		sign = INTEGER(getListElement(params, "sign"))[0];
	}
    virtual ~LCPdistance(){}
	virtual double distance(const int&is, const int& js);
};

#endif
