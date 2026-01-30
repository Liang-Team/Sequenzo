#ifndef DHDDISTANCECALCULATOR_H
#define DHDDISTANCECALCULATOR_H
#include "distancecalculator.h"

class DHDdistance: public DistanceCalculator{
  protected:
    double * scost;
    int alphasize;
	double maxdist;
	
  public:
	DHDdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS)
		:DistanceCalculator(normS, Ssequences, seqdim, lenS), scost(NULL), alphasize(0), maxdist(0){}
	DHDdistance(DHDdistance *dc): DistanceCalculator(dc),  scost(dc->scost), alphasize(dc->alphasize), maxdist(dc->maxdist){}
	virtual DistanceCalculator* copy(){return new DHDdistance(this);}
	virtual void setParameters(SEXP params){
		scost = REAL(getListElement(params, "scost"));
		alphasize = INTEGER(getListElement(params, "alphasize"))[0];
		maxdist = REAL(getListElement(params, "maxdist"))[0];
	}
    virtual ~DHDdistance(){}
    virtual double distance(const int&is, const int& js);
};

#endif
