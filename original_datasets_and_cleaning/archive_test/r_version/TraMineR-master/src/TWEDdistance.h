#ifndef TWEDDISTANCECALCULATOR_H
#define TWEDDISTANCECALCULATOR_H
#include "distancecalculator.h"
#include "OMdistance.h"

class TWEDdistance: public OMdistance{
  protected:
    double nu;
	double lambda;
	
  public:
	TWEDdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	TWEDdistance(TWEDdistance *dc);
	virtual DistanceCalculator * copy(){return new TWEDdistance(this);}
	virtual void setParameters(SEXP params);
    virtual ~TWEDdistance();
    virtual double distance(const int&is, const int& js);
};


#endif
