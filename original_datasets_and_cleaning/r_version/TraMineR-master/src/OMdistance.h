#ifndef OMDISTANCECALCULATOR_H
#define OMDISTANCECALCULATOR_H
#include "distancecalculator.h"

class OMdistance: public DistanceCalculator{
  protected:
	double * fmat;
    double * scost;
    int alphasize;
    double indel;
	int fmatsize;
	double maxscost;
	
  public:
	OMdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	OMdistance(OMdistance *dc);
	virtual DistanceCalculator* copy(){return new OMdistance(this);}
	virtual void setParameters(SEXP params);
    virtual ~OMdistance();
    virtual double distance(const int&is, const int& js);
};


#endif
