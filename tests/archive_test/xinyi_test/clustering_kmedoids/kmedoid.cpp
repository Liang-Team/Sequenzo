#include "kmedoid.h"

KMedoid::KMedoid(SEXP Snelement, SEXP diss, SEXP _expr,SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist): 
			KMedoidBase(Snelement, diss, _expr, _rho, Scentroids, Snpass, Sweights, Sisdist){
				TMRLOG(1, "Start KMedoid builded\n");
				saved= new int[nelements];
				clusterMembership = new int[nelements*nclusters];
				clusterSize = new int[nclusters];
				//	centroids = (int*) R_alloc(nclusters,sizeof(int));
				//errors = new double[nclusters];
				TMRLOG(1, "KMEDOID builded\n");
}
					


#define KMEDOIDBODY_INCLUDED
	//Including version based on dist object
	#define DISTOBJECT_VERSION
		#include "kmedoidbody.cpp"
	#undef DISTOBJECT_VERSION

	//Including version based on a distance matrix
	#define DISTMATRIX_VERSION
		#include "kmedoidbody.cpp"
	#undef DISTMATRIX_VERSION

#undef KMEDOIDBODY_INCLUDED
