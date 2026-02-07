#ifndef OMVIDISTANCECALCULATOR_H
#define OMVIDISTANCECALCULATOR_H
#include "OMdistance.h"

class OMVIdistance: public OMdistance{
	class IndelCalculator{
		public:
			IndelCalculator(){}
			virtual ~IndelCalculator(){}
			virtual double getIndel(const int&state, const int& prev, const int& next)=0;
			virtual IndelCalculator * copy()=0;
	};
	class VaryingIndelCalculator: public IndelCalculator{
		double * indellist;
		public:
			VaryingIndelCalculator(double * _indellist):indellist(_indellist){}
			VaryingIndelCalculator(VaryingIndelCalculator * vic):indellist(vic->indellist){}
			virtual ~VaryingIndelCalculator(){}
			virtual IndelCalculator * copy(){return new VaryingIndelCalculator(this);}
			virtual double getIndel(const int&state, const int& prev, const int& next){
				return indellist[state];
			}
	};
	class OMlocIndelCalculator: public IndelCalculator{
		protected:
			double timecost;
			double localcost;
			double * scost;
			int alphasize;
		public:
			OMlocIndelCalculator(const double & tc, const double &lc, double * sc, const int& as):timecost(tc), localcost(lc), scost(sc), alphasize(as){
			}
			OMlocIndelCalculator(OMlocIndelCalculator *vis):timecost(vis->timecost), localcost(vis->localcost), scost(vis->scost), alphasize(vis->alphasize){
			}
			virtual ~OMlocIndelCalculator(){}
			virtual IndelCalculator * copy(){return new OMlocIndelCalculator(this);}
			virtual double getIndel(const int&state, const int& prev, const int& next){
				TMRLOG(10, "SCostP %g, SCostN=%g, total=%g", scost[MINDICE(prev,state,alphasize)], scost[MINDICE(next,state,alphasize)],
					timecost+localcost*(scost[MINDICE(prev,state,alphasize)]+ scost[MINDICE(next,state,alphasize)])/2);
				return timecost+localcost*(scost[MINDICE(prev,state,alphasize)]+ scost[MINDICE(next,state,alphasize)])/2;
			}
	};
	class OMlocIndelCalculatorMin: public OMlocIndelCalculator{
		public:
			OMlocIndelCalculatorMin(const double & tc, const double &lc, double * sc, const int& as):OMlocIndelCalculator(tc, lc, sc, as){
			}
			OMlocIndelCalculatorMin(OMlocIndelCalculatorMin * vis):OMlocIndelCalculator(vis){
			}
			virtual ~OMlocIndelCalculatorMin(){}
						virtual IndelCalculator * copy(){return new OMlocIndelCalculatorMin(this);}
			virtual double getIndel(const int&state, const int& prev, const int& next){
				TMRLOG(10, "SCostP %g, SCostN=%g, total=%g", scost[MINDICE(prev,state,alphasize)], scost[MINDICE(next,state,alphasize)],
					timecost+localcost*fmin2(scost[MINDICE(prev,state,alphasize)], scost[MINDICE(next,state,alphasize)]));
				return timecost+localcost*fmin2(scost[MINDICE(prev,state,alphasize)], scost[MINDICE(next,state,alphasize)]);
			}
	};
  protected:
	IndelCalculator * indelCalc;
	
	
  public:
	OMVIdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS);
	OMVIdistance(OMVIdistance *dc);
	virtual DistanceCalculator* copy(){return new OMVIdistance(this);}
	virtual void setParameters(SEXP params);
    virtual ~OMVIdistance();
    virtual double distance(const int&is, const int& js);
	double getIndel(const int&state, const int& prev, const int& next);
};

inline double OMVIdistance::getIndel(const int&state, const int& prev, const int& next){
	return this->indelCalc->getIndel(state, prev, next);
	// switch(indelmethod){
		// case 0:
			// return indellist[state];
		// case 1: //Localized OM
			// return timecost+localcost*(scost[MINDICE(prev,state,alphasize)]+ scost[MINDICE(next,state,alphasize)])/2;
		// case 2: 
			// return timecost+localcost*fmin2(scost[MINDICE(prev,state,alphasize)], scost[MINDICE(next,state,alphasize)]);
	// }
	// return indel;
}

#endif
