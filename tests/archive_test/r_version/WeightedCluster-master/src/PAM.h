#ifndef PAM_H_INCLUDED
#define PAM_H_INCLUDED
#include "kmedoidbase.h"

class PAM: public KMedoidBase{
	protected:
		double * dysmb;
	public:
		PAM(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist);
		virtual ~PAM(){
			delete[] dysmb;
		}
		virtual double runclusterloop(const int & ipass);
		virtual double runclusterloop_dist(const int & ipass);
		
};

#endif
