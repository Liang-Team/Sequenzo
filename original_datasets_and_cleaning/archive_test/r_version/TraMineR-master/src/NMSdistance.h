#ifndef NMSDISTANCECALCULATOR_H
#define NMSDISTANCECALCULATOR_H
#include "distancecalculator.h"

class SUBSEQdistance: public DistanceCalculator{
	protected:
		double *selfmatvect;
		double *kvect;
		double *kweights;
		int distMethod;
		int distTransform;
	public:
		SUBSEQdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS):DistanceCalculator(normS, Ssequences, seqdim, lenS),kweights(NULL), distMethod(0){
			selfmatvect= new double[nseq*maxlen];
			kvect =  new double[maxlen];
		}
		SUBSEQdistance(SUBSEQdistance *dc):DistanceCalculator(dc),kweights(dc->kweights), distMethod(dc->distMethod){
			selfmatvect= new double[nseq*maxlen];
			memcpy(this->selfmatvect, dc->selfmatvect, nseq*maxlen*sizeof(double));
			kvect =  new double[maxlen];
			
		}
		//virtual DistanceCalculator* copy(){return new SUBSEQdistance(this);}
		virtual ~SUBSEQdistance(){
			delete[] selfmatvect;
			delete[] kvect;
		}
		void resetKvect(){
			for(int k=0;k <maxlen;k++){
				this->kvect[k]=0.0;
			}
		}
		virtual void computeattr(const int&is, const int& js)=0;
		virtual void setParameters(SEXP params);
		double distance(const int&is, const int& js);
};


class NMSdistance: public SUBSEQdistance{
  protected:
    int zmatsize;
	double *hmat;
	double *vmat;
	int *zmat;
	
	
  public:
	NMSdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);//:	SUBSEQdistance( normS, Ssequences, seqdim, lenS){}
	NMSdistance(NMSdistance * dc);
	virtual DistanceCalculator* copy(){return new NMSdistance(this);}
    virtual ~NMSdistance();
	void computeattr(const int&is, const int& js);
};

#endif
