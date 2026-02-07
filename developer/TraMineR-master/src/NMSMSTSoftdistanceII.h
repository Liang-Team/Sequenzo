#ifndef NMSMSTSoftdistanceIICALCULATOR_H
#define NMSMSTSoftdistanceIICALCULATOR_H
#include "NMSdistance.h"

class NMSMSTSoftdistanceII: public SUBSEQdistance{
  protected:
	double *e;
	double *e1;
	int rowsize;
	double * softmatch;
	int alphasize;
	
  public:
  	NMSMSTSoftdistanceII(NMSMSTSoftdistanceII *dc);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
  	NMSMSTSoftdistanceII(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
	virtual DistanceCalculator* copy(){return new NMSMSTSoftdistanceII(this);}
	//NMSMSTdistance(const int& pnorm, int * psequences, const int & pnseq,  int * pslen, const int &pmaxlen, double *pseqdur);
    virtual ~NMSMSTSoftdistanceII();
	virtual void setParameters(SEXP params);
	void computeattr(const int&is, const int& js);
};

#endif
