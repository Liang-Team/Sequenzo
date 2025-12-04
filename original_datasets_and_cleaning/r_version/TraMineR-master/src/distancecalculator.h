#ifndef DISTANCECALCULATOR_H
#define DISTANCECALCULATOR_H
#include "TraMineR.h"



class DistanceCalculator{
protected:
	int norm;
	int * sequences;
	int nseq;
	int * slen;
	int maxlen;
public:
	//INTEGER(normS)[0], INTEGER(Ssequences), INTEGER(seqdim)[0], INTEGER(lenS), INTEGER(seqdim)[1]
    DistanceCalculator(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS):norm(INTEGER(normS)[0]), sequences(INTEGER(Ssequences)), nseq(INTEGER(seqdim)[0]), slen(INTEGER(lenS)), maxlen(INTEGER(seqdim)[1]){
	}
	DistanceCalculator(DistanceCalculator *dc):norm(dc->norm), sequences(dc->sequences), nseq(dc->nseq), slen(dc->slen), maxlen(dc->maxlen){
	}
    virtual ~DistanceCalculator() {}
    virtual double distance(const int&is, const int& js)=0;
    virtual void setParameters(SEXP params)=0;
	virtual DistanceCalculator* copy()=0;
	double normalizeDistance(const double& rawdist, const double& maxdist, const int & l1, const int & l2);
	double normalizeDistance(const double& rawdist, const double& maxdist, const double & l1, const double & l2);
	static void finalizeDistanceCalculator(SEXP ptr);
	static SEXP distanceCalculatorFactory(DistanceCalculator *ds);
	SEXP getListElement(SEXP list, const char *str) {
		int i;
		SEXP elmt = R_NilValue, names = Rf_getAttrib(list, R_NamesSymbol);
		for (i = 0; i < Rf_length(list); i++) {
			if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
				elmt = VECTOR_ELT(list, i);
				break;
			}
		}
		return elmt;
	}
};

//inline double DistanceCalculator::normalizeDistance(const double& rawdist, const double& maxdist, const double & l1, const double & l2) {
//	return this->normalizeDistance(rawdist, maxdist, (double)l1, (double) l2);
//}

inline double DistanceCalculator::normalizeDistance(const double& rawdist, const double& maxdist, const double & l1, const double & l2) {
    if (rawdist==0)return 0;
    switch (norm) {
    case 0:
        return rawdist;
    case 1:
        if (l1>l2)return rawdist/((double)l1);
        else if (l2>0) return rawdist/((double)l2);
        return 0;
    case 2:
        if (l1*l2==0) {
            if (l1!=l2)return 1;
            return 0;
        }
		//R_pow(l1, 0.5)* R_pow(l2, 0.5) reduce the probability of exceding DOUBLE_MAX
        return 1-((maxdist-rawdist)/(2*R_pow(l1, 0.5)* R_pow(l2, 0.5)));
    case 3:
        if (maxdist==0)return 1;
        return rawdist/maxdist;
	case 4:
		if (maxdist==0)return 1;
        return (2*rawdist)/(rawdist+maxdist);
		
    }
    return rawdist;
}


inline SEXP DistanceCalculator::distanceCalculatorFactory(DistanceCalculator *ds) {
    SEXP SDO, classname;
	PROTECT(classname = Rf_allocVector(STRSXP, 1));
	SET_STRING_ELT(classname, 0, Rf_mkChar("DistanceCalculator"));
    SDO = R_MakeExternalPtr(ds, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(SDO, (R_CFinalizer_t) finalizeDistanceCalculator, TRUE);
    Rf_classgets(SDO, classname);
	UNPROTECT(1);
    return SDO;
}

#endif
