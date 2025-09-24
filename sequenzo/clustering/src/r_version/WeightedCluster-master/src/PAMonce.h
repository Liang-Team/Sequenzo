#ifndef PAMONCE_H_INCLUDED
#define PAMONCE_H_INCLUDED
#include "PAM.h"

#define WEIGHTED_CLUST_TOL -1e-10
class PAMonce: public PAM{
	protected:
		double * fvect;
	public:
		PAMonce(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights ,SEXP Sisdist);
		virtual ~PAMonce(){
			delete[] fvect;
		}
		virtual double runclusterloop(const int & ipass);
		virtual double runclusterloop_dist(const int & ipass);

		
};

#endif
