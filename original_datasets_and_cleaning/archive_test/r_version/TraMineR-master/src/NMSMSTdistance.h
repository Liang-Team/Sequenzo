#ifndef NMSMSTDISTANCECALCULATOR_H
#define NMSMSTDISTANCECALCULATOR_H
#include "NMSdistance.h"

class NMSMSTdistance: public SUBSEQdistance{
  protected:
	double *e;
	double *e1;
	double *t;
	double *t1;
	int rowsize;
	double * seqdur;
	
  public:
  	NMSMSTdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
  	NMSMSTdistance(NMSMSTdistance* dc);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
	virtual DistanceCalculator* copy(){return new NMSMSTdistance(this);}
	//NMSMSTdistance(const int& pnorm, int * psequences, const int & pnseq,  int * pslen, const int &pmaxlen, double *pseqdur);
    virtual ~NMSMSTdistance();
	virtual void setParameters(SEXP params);
	void computeattr(const int&is, const int& js);
};

#endif
