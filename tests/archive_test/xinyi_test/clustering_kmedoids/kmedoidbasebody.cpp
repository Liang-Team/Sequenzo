#ifdef KMEDOIDBASEBODY_INCLUDED

#if !defined(DISTOBJECT_VERSION) && !defined(DISTMATRIX_VERSION)
	#error KMEDOIDBASEBODY version not defined
#endif

#include "kmedoidbase.h"


#ifdef DISTMATRIX_VERSION
	void KMedoidBase::getrandommedoids(){
#else
	void KMedoidBase::getrandommedoids_dist(){
#endif
	int i=0, j;
	int * rndvect;
	//int counter=0;
	SEXP randomCentroid;
	/* Copy values */
	while(i<nclusters){
		
		PROTECT(randomCentroid = Rf_eval(expr, rho));
		rndvect = INTEGER(randomCentroid);
		
		//Now checking that this is a correct solution
		for (i = 0; i < nclusters; i++) { 
			centroids[i] = rndvect[i];
			j = i+1;
			while(j < nclusters && 
					#ifdef DISTMATRIX_VERSION
						distmatrix[MINDICE(centroids[i],rndvect[j],nelements)]>0) { 
					#else
						distmatrix[TMRDISTINDEX(centroids[i]+1, rndvect[j]+1, nelements)]>0) {  //TMRDISTINDEX is R indice based
					#endif

				j++;
			}
			if(j<nclusters){
				break;
			}
			//counter++;
		}
		UNPROTECT(1);
	}
/* 	for (int icluster = 0; icluster < nclusters; icluster++){
		REprintf("%d, ", centroids[icluster]);
	}
	REprintf("\n"); */
	return;
}
#ifdef DISTMATRIX_VERSION
	void KMedoidBase::computeMaxDist(){
		int ij=0, i=0, j=0;
		for (i = 0; i < nelements; i++){
			ij = i*nelements;
			for (j = i+1; j < nelements; j++) {
				if(distmatrix[ij+j]>maxdist){
					maxdist = distmatrix[ij+j];
				}
			}
		}
		maxdist = 1.1*maxdist+1.0;
	}
#else
	void KMedoidBase::computeMaxDist_dist(){
		int ij=0;
		for (ij = 0; ij < distlength; ij++){
			if(distmatrix[ij]>maxdist){
				maxdist = distmatrix[ij];
			}
		}
		maxdist = 1.1*maxdist+1.0;
	}
#endif
	
#ifdef DISTMATRIX_VERSION
	void KMedoidBase::buildInitialCentroids(){
#else
	void KMedoidBase::buildInitialCentroids_dist(){
#endif
	/* compute beter[i] for all non-representatives:
	     * also find ammax := max_{..} and nmax := argmax_i{beter[i]} ... */
	int nmax = -1; /* -Wall */
	double ammax, cmd;
	int ij=0, i=0, j=0;
	#ifdef DISTMATRIX_VERSION
		this->computeMaxDist();
		TMRLOG(1, "MaxDist matrix computed=%g\n", maxdist);
	#else
		this->computeMaxDist_dist();
		TMRLOG(1, "MaxDist dist computed=%g\n", maxdist);
	#endif
	for (i = 0; i < nelements; i++){
		dysma[i] = maxdist;
		clusterid[i] = 0;
	}
	for (int k = 0; k < nclusters; ++k) {
	    ammax = 0.;
	    for(i = 0; i < nelements; i++) {
			if (clusterid[i] == 0) {
				#ifdef DISTMATRIX_VERSION
				double beter = 0.;
				ij=i*nelements;
				for (j = 0; j < nelements; j++) {
					cmd = dysma[j] - distmatrix[ij+j];
					//if(j==i-1){
						//TMRLOG(10, "i=%d, j=%d, dist=%g, cmd=%g", i, j, distmatrix[ij+j], cmd);
					//}
					if (cmd > 0.) {
						beter += weights[j]*cmd;
					}
				}
				#else
				//The dist loop doesn't take into account i==j
				double beter = weights[i]* dysma[i];
				ij =DL_FIRST_INIT(i, nelements);
				for(j=0;j <i;j++){
					ij += DL_FIRST_INC(i, j, nelements);
					cmd = dysma[j] - distmatrix[DL_FIRST_ACCESS(ij, i, j)];
					//if(j==i-1){
						//TMRLOG(10, "i=%d, j=%d, distindice=%d, dist=%g, cmd=%g", i, j, DL_FIRST_ACCESS(ij, i, j), distmatrix[DL_FIRST_ACCESS(ij, i, j)], cmd);
					//}
					if (cmd > 0.) {
						beter += weights[j]*cmd;
					}
				}
				ij = DL_SEC_INIT(i, nelements);
				for(j=i+1;j <nelements;j++){
					cmd = dysma[j] - distmatrix[DL_SEC_ACCESS(ij, i, j)];
					// if(j==i+1){
						//TMRLOG(10, "i=%d, j=%d, distindice=%d, dist=%g, cmd=%g", i, j, DL_SEC_ACCESS(ij, i, j), distmatrix[DL_SEC_ACCESS(ij, i, j)], cmd);
					// }
					if (cmd > 0.) {
						beter += weights[j]*cmd;
					}
				}
				#endif
				if (ammax <= beter) {
					/*  does < (instead of <= ) work too? -- NO! */
					ammax = beter;
					nmax = i;
					TMRLOG(10, "%d choosen as centroid %g", nmax, ammax);
				}
			}
	    }
	    
	    clusterid[nmax] = 1;/* = .true. : found new representative */
		centroids[k] = nmax;
		//REprintf("%d,\n", nmax);

	    /* update dysma[] : dysma[j] = D(j, nearest_representative) */
		#ifdef DISTMATRIX_VERSION
	    for (j = 0; j < nelements; ++j) {
			ij = MINDICE(nmax,j,nelements);
			if (dysma[j] > distmatrix[ij]){
				dysma[j] = distmatrix[ij];
			}
	    }
		#else
		ij =DL_FIRST_INIT(nmax, nelements);
		for(j=0;j <nmax;j++){
			ij += DL_FIRST_INC(nmax, j, nelements);
			if (dysma[j] > distmatrix[DL_FIRST_ACCESS(ij, nmax, j)]){
				dysma[j] = distmatrix[DL_FIRST_ACCESS(ij, nmax, j)];
			}
		}
		dysma[nmax]=0;
		ij = DL_SEC_INIT(nmax, nelements);
		for(j=nmax+1;j <nelements;j++){
			if (dysma[j] > distmatrix[DL_SEC_ACCESS(ij, nmax, j)]){
				dysma[j] = distmatrix[DL_SEC_ACCESS(ij, nmax, j)];
			}
		}
		#endif
	}

	/* for (int icluster = 0; icluster < nclusters; icluster++){
		REprintf("%d, ", centroids[icluster]);
	}
	REprintf("\n"); */
	return;
}

#endif
