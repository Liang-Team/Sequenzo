#ifndef KMEDOIDBASE_H_INCLUDED
#define KMEDOIDBASE_H_INCLUDED
#include "cluster.h"

#define DERRORTOT 0
#define DIFOUND 1
#define DMETHOD 2
class KMedoidBase{
	protected:
		int nclusters;
		int nelements;
		double* distmatrix;
		int npass;
		int *clusterid;
		double* stat;
		SEXP expr; 
		SEXP exprCentroid;
		SEXP rho;
		double * weights;
		int* centroids;
		int buildMethod;
		SEXP ans;
		//Clustering variable
		int * tclusterid;
		
		//Building variable
		double * dysma;
		double maxdist;
		int isdist;
		int distlength;
	public:
		KMedoidBase(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist);
		virtual ~KMedoidBase(){
			delete[] dysma;
			delete[] tclusterid;
			delete[] centroids;
		}
		virtual void clean();
		virtual SEXP getClustering(){
			return ans;
		}
		void findCluster();
		
		// Variant for dist and non dist
		void getrandommedoids();
		void buildInitialCentroids();
		void computeMaxDist();
		virtual double runclusterloop(const int & ipass)=0;
		
		
		void getrandommedoids_dist();
		void buildInitialCentroids_dist();
		void computeMaxDist_dist();
		virtual double runclusterloop_dist(const int & ipass)=0;
};

#endif
