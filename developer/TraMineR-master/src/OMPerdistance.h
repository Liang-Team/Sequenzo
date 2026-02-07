#ifndef OMPERDISTANCECALCULATOR_H
#define OMPERDISTANCECALCULATOR_H
#include "OMdistance.h"

class OMPerdistance: public OMdistance{
	double timecost;
	double * seqdur;
	double * indellist;
	int * seqlen;
  public:
	OMPerdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	OMPerdistance(OMPerdistance *dc);
	virtual void setParameters(SEXP params);
    virtual ~OMPerdistance();
    virtual double distance(const int&is, const int& js);
	inline double getIndel(const int& indice, const int& state){
		return this->indellist[state]+timecost*(seqdur[indice]);
	}
	virtual DistanceCalculator* copy(){return new OMPerdistance(this);}
	inline double getSubCost(const int& i_state, const int& j_state, const int& i_state_indice, const int& j_state_indice){
		
		if(i_state==j_state){
			double diffdur= seqdur[i_state_indice]-seqdur[j_state_indice];
			if(diffdur<0) {
				return -1.0*(timecost*diffdur);
			}else{
				return(timecost*diffdur);
			}
		}else{
			//double commondur =fmin2(seqdur[i_state_indice], seqdur[j_state_indice]);
			//return(commondur*scost[MINDICE(i_state, j_state, alphasize)] + diffdur*timecost);
			return(scost[MINDICE(i_state, j_state, alphasize)] + (seqdur[i_state_indice]+seqdur[j_state_indice])*timecost);
		}
	}
};

#endif
