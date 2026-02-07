#ifndef DISTANCEOBJECT_H
#define DISTANCEOBJECT_H
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
//using namespace std;
//#define TMRDISTINDEX(i,j,n) (n*(i-1) - i*(i-1)/2 + j-i-1)

//Définition de type pour aleger le code
//Une map sur le type d'événements, et la classe événement
class DistanceObject{
protected:
  int* magicIndex;
  int * magicSeq;
  int finalnseq;
  SEXP ans;
  double * result;
  
public:
    DistanceObject(SEXP magicIndexS, SEXP magicSeqS);
    ~DistanceObject();
    void setDistance(const int &is,const int &js, const double& cmpres);
    static inline int distIndex(const int &i,const int &j,const int &n){
		if(i<j)return TMRDISTINDEX(i,j,n);
		else return TMRDISTINDEX(j,i,n);
    }
    SEXP getDistObject(){return ans;}

};

DistanceObject::DistanceObject(SEXP magicIndexS, SEXP magicSeqS){
  this->magicIndex=INTEGER(magicIndexS);
  this->magicSeq=INTEGER(magicSeqS);
  this->finalnseq=Rf_length(magicSeqS);
  PROTECT(ans = Rf_allocVector(REALSXP, (finalnseq*(finalnseq-1)/2)));
  result=REAL(ans);
}
DistanceObject::~DistanceObject(){}

inline void DistanceObject::setDistance(const int &is,const int &js, const double& cmpres){
	int j_start=magicIndex[js];
	int j_end=magicIndex[js+1];
	int i_start=magicIndex[is];
	int i_end=magicIndex[is+1];
	int i_index, j_index, i, j, base_index;
	for(i=i_start;i<i_end;i++){
		i_index=magicSeq[i];
		//n*(i-1) - i*(i-1)/2 + j-i
		for(j=j_start; j<j_end; j++) {
			j_index=magicSeq[j];
			if(i_index!=j_index) {
				base_index=distIndex(i_index,j_index,finalnseq);
//				REprintf("Unique (%d,%d) => (%d,%d)(%d) => %f \n",is,js,i_index,j_index,(base_index),cmpres);
				result[base_index]=cmpres;
			}
		}
	}
}


void finalizeDistanceObject(SEXP ptr){
	DistanceObject * sdo;
	sdo= static_cast<DistanceObject *>(R_ExternalPtrAddr(ptr));
	delete sdo;
}
inline SEXP distanceObjectFactory(DistanceObject *ds) {
    SEXP SDO, classname;
	PROTECT(classname = Rf_allocVector(STRSXP, 1));
	SET_STRING_ELT(classname, 0, Rf_mkChar("DistanceObject"));
    SDO = R_MakeExternalPtr(ds, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(SDO, (R_CFinalizer_t) finalizeDistanceObject, TRUE);
    Rf_classgets(SDO, classname);
	UNPROTECT(1);
    return SDO;
}

#endif // DISTANCEOBJECT_H
