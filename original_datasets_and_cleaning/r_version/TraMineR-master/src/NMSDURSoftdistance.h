#ifndef NMSDURSOFTDISTANCECALCULATOR_H
#define NMSDURSOFTDISTANCECALCULATOR_H
#include "NMSdistance.h"

class NMSDURSoftdistance: public SUBSEQdistance{
  protected:
	double *e;
	double *e1;
	double *t_i;
	double *t1_i;
	double *t1_j;
	double *t_j;
	double *t_ij;
	int rowsize;
	double * seqdur;
	double * softmatch;
	int alphasize;
	
  public:
  	NMSDURSoftdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
  	NMSDURSoftdistance(NMSDURSoftdistance * dc);//:SUBSEQdistance(normS, Ssequences, seqdim, lenS), seqdur(NULL){}
	virtual DistanceCalculator* copy(){return new NMSDURSoftdistance(this);}
	//NMSDURdistance(const int& pnorm, int * psequences, const int & pnseq,  int * pslen, const int &pmaxlen, double *pseqdur);
    virtual ~NMSDURSoftdistance();
	virtual void setParameters(SEXP params);
	void computeattr(const int&is, const int& js);
};

#endif
