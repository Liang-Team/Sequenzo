#ifndef NMSMSTSOFTDISTANCECALCULATOR_H
#define NMSMSTSOFTDISTANCECALCULATOR_H
#include "NMSdistance.h"

class NMSMSTSoftdistance: public SUBSEQdistance{
  protected:
	double *e;
	double *e1;
	double *t;
	double *t1;
	int rowsize;
	double * seqdur;
	double * softmatch;
	int alphasize;
	
  public:
  	NMSMSTSoftdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
  	NMSMSTSoftdistance(NMSMSTSoftdistance *dc);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
	virtual DistanceCalculator* copy(){return new NMSMSTSoftdistance(this);}
	//NMSMSTdistance(const int& pnorm, int * psequences, const int & pnseq,  int * pslen, const int &pmaxlen, double *pseqdur);
    virtual ~NMSMSTSoftdistance();
	virtual void setParameters(SEXP params);
	void computeattr(const int&is, const int& js);
};

#endif
