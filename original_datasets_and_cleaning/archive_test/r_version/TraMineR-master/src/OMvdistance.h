#ifndef OMvdistanceCALCULATOR_H
#define OMvdistanceCALCULATOR_H
#include "OMdistance.h"

class OMvdistance: public OMdistance{
	double * seqdur;
	double * indellist;
	int sublink;
  public:
	OMvdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	OMvdistance(OMvdistance *dc);
	virtual DistanceCalculator* copy(){return new OMvdistance(this);}
	virtual void setParameters(SEXP params);
    virtual ~OMvdistance();
    virtual double distance(const int&is, const int& js);
	inline double getIndel(const int& indice, const int& state){
		return this->indellist[state]*seqdur[indice];
	}
  inline double getSubCost(const int& i_state, const int& j_state, const int& i_state_indice, const int& j_state_indice){
    if (sublink==1) {
      return(scost[MINDICE(i_state, j_state, alphasize)] * (seqdur[i_state_indice]+seqdur[j_state_indice]));
    }
    return(scost[MINDICE(i_state, j_state, alphasize)] * sqrt(seqdur[i_state_indice]*seqdur[j_state_indice]));
  }
};

#endif
