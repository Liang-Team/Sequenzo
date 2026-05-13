#include "kmedoidbase.h"

KMedoidBase::KMedoidBase(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist): 
					nclusters(Rf_length(Scentroids)), nelements(INTEGER(Snelement)[0]), distmatrix(REAL(diss)),
					npass(INTEGER(Snpass)[0]),clusterid(NULL), stat(NULL), expr(_expr), rho(_rho), 
					weights(REAL(Sweights)), centroids(NULL), maxdist(0), isdist(INTEGER(Sisdist)[0]){
						TMRLOG(1, "Start KMedoidBase builded nclusters=%d nelements=%d\n", nclusters, nelements);
						distlength=(nelements*(nelements-1))/2;
						SEXP cluster, statS;
						PROTECT(ans = Rf_allocVector(VECSXP, 2));
						PROTECT(cluster = Rf_allocVector(INTSXP, nelements));
						PROTECT(statS = Rf_allocVector(REALSXP, 3));
						SET_VECTOR_ELT(ans, 0, cluster);
						SET_VECTOR_ELT(ans, 1, statS);
						TMRLOG(1, "Return memory allocated\n", nclusters, nelements);
						clusterid = INTEGER(cluster);
						tclusterid = new int[nelements];
						for (int i = 0; i < nelements; i++) { 
							clusterid[i]=-1;
							tclusterid[i]=-1;
						}
						this->stat = REAL(statS);
						stat[DERRORTOT] = DBL_MAX;
						stat[DIFOUND] = -1;
						TMRLOG(1, "Return memory initialized\n", nclusters, nelements);
						int *centroidsBegin = INTEGER(Scentroids);
						this->centroids= new int[nclusters];
						for (int i = 0; i < nclusters; i++) { 
							centroids[i] = centroidsBegin[i];
						}
						TMRLOG(1, "Centroid finished\n", nclusters, nelements);

						dysma= new double[nelements];
						TMRLOG(1, "KMedoidBase builded\n");
					}
					
void KMedoidBase::clean(){
	UNPROTECT(3);
}

void KMedoidBase::findCluster(){
	TMRLOG(1, "Finding clusters, dist=%d\n", this->isdist);
	int i, j;
	int ipass=0;
	double method=0;
	do {
		R_CheckUserInterrupt();
		
		if(npass!=0){ // To do allow specify start solution & (npass!=0)) {
			if(ipass>0){
				TMRLOG(1, "Random medoids, dist=%d\n", this->isdist);
				
				if(this->isdist){
					this->getrandommedoids_dist();
				}else{
					this->getrandommedoids();
				}
				method=3.0;
			} 
			else{
				TMRLOG(1, "PAM centroids, dist=%d\n", this->isdist);
				
				if(this->isdist){
					this->buildInitialCentroids_dist();
				}else{
					this->buildInitialCentroids();
				}
				
				method=1.0;
			}
		}else{
		
			if(this->isdist){
				this->computeMaxDist_dist();
			}
			else {
				this->computeMaxDist();
			}
			method = 0.0;
			
		}
		TMRLOG(1, "Starting loop\n");
		double total=0;
		if(this->isdist){
			total = this->runclusterloop_dist(ipass);
		}else {
			total = this->runclusterloop(ipass);
		}
	
		TMRLOG(1, "Finished loop\n");
		if(ipass==0){
			for (j = 0; j < nelements; j++) {
				clusterid[j] = centroids[tclusterid[j]];
			}
			stat[DIFOUND] = 1;
			stat[DERRORTOT] = total;
			stat[DMETHOD] = method;
		} else {
			for (i = 0; i < nelements; i++) { 
				if (clusterid[i]!=centroids[tclusterid[i]]) { 
					if (total < stat[DERRORTOT]) { 
						//REprintf("Current best\n", *error, total);
						stat[DIFOUND] = 1;
						stat[DERRORTOT] = total;
						stat[DMETHOD] = method;
						/* Replace by the centroid in each cluster. */
						for (j = 0; j < nelements; j++) {
							clusterid[j] = centroids[tclusterid[j]];
						}
					}
					break;
				}
			}
			if (i==nelements) { 
				//REprintf("Same Solution\n", *error, total);
				stat[DIFOUND]++; /* break statement not encountered */
			}
		}
	} while (++ipass < npass);
	return;
}


#define KMEDOIDBASEBODY_INCLUDED
	//Including version based on dist object
	#undef DISTMATRIX_VERSION
	#define DISTOBJECT_VERSION
		#include "kmedoidbasebody.cpp"
	#undef DISTOBJECT_VERSION

	//Including version based on a distance matrix
	#define DISTMATRIX_VERSION
		#include "kmedoidbasebody.cpp"
	#undef DISTMATRIX_VERSION

#undef KMEDOIDBASEBODY_INCLUDED
