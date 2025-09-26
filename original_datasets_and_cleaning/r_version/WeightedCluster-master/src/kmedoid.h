#ifndef KMEDOID_H_INCLUDED
#define KMEDOID_H_INCLUDED
#include "kmedoidbase.h"

class KMedoid: public KMedoidBase{
	protected:
		int * saved;
		int* clusterMembership;
		int* clusterSize;
		//double* errors;
	public:
		KMedoid(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist);
		virtual ~KMedoid(){
			delete [] saved;
			delete [] clusterMembership;
			delete [] clusterSize;
		}
		virtual double runclusterloop(const int & ipass);
		virtual double runclusterloop_dist(const int & ipass);
		virtual void getclustermedoids();
		virtual void getclustermedoids_dist();
};

#endif
