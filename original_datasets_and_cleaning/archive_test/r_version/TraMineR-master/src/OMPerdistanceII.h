#ifndef OMPERDISTANCEIICALCULATOR_H
#define OMPERDISTANCEIICALCULATOR_H
#include "OMdistance.h"

class OMPerdistanceII: public OMdistance{
	double timecost;
	double * seqdur;
	double * indellist;
	double * tokdeplist;
	int * seqlen;
  public:
	OMPerdistanceII(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	OMPerdistanceII(OMPerdistanceII *dc);
	virtual void setParameters(SEXP params);
    virtual ~OMPerdistanceII();
    virtual double distance(const int&is, const int& js);
	inline double getIndel(const int& indice, const int& state){
		return this->indellist[state]+timecost*tokdeplist[state]*(seqdur[indice]);
	}
	virtual DistanceCalculator* copy(){return new OMPerdistanceII(this);}
	inline double getSubCost(const int& i_state, const int& j_state, const int& i_state_indice, const int& j_state_indice){
		
		if(i_state==j_state){
			double diffdur= seqdur[i_state_indice]-seqdur[j_state_indice];
			if(diffdur<0) {
				return -1.0*(timecost*diffdur*tokdeplist[i_state]);
			}else{
				return(timecost*diffdur*tokdeplist[i_state]);
			}
		}else{
			//double commondur =fmin2(seqdur[i_state_indice], seqdur[j_state_indice]);
			//return(commondur*scost[MINDICE(i_state, j_state, alphasize)] + diffdur*timecost);
			return(scost[MINDICE(i_state, j_state, alphasize)] + (tokdeplist[i_state]*seqdur[i_state_indice]+tokdeplist[j_state]*seqdur[j_state_indice])*timecost);
		}
	}
};

#endif
