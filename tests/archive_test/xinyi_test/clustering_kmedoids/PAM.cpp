#include "PAM.h"

PAM::PAM(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist):
			KMedoidBase(Snelement, diss, _expr, _rho, Scentroids, Snpass, Sweights, Sisdist){
				TMRLOG(1, "Start PAM builded\n");
	this->dysmb= new double[nelements];
	TMRLOG(1, "PAM builded\n");
}
 double PAM::runclusterloop_dist(const int & ipass){
	Rf_error("[!] Not (yet) implemented (please use PAMonce algorithm)\n");
	return DBL_MAX;
 }
 double PAM::runclusterloop(const int & ipass){
	TMRLOG(1, "PAM LOOP\n");
	double dzsky;
	int hbest = -1, nbest = -1;/* init: -Wall*/
	int i, j, k, ij, icluster, h;
	double total=-1.0;
	TMRLOG(1,"Centroids lists (in):");
	for (icluster = 0; icluster < nclusters; icluster++){
		TMRLOG(1,"%d, \n", centroids[icluster]);
	}
/* ====== second algorithm: SWAP. ====== */

	/* Hmm: In the following, we RE-compute dysma[];
	 *      don't need it first time; then only need *update* after swap */

	/*--   Loop : */
	//total=-1;
	do{
		TMRLOG(1, "BUILD dysma, dysmb\n");
		//Here computing dysma (closest med) and dysmb second med
		for (i = 0; i < nelements; i++) {
			/*  dysma[j] := D_j  d(j, <closest medi>)  [KR p.102, 104]
			 *  dysmb[j] := E_j  d(j, <2-nd cl.medi>)  [p.103] */
			dysma[i] = maxdist;
			dysmb[i] = maxdist;
			ij=i*nelements;
			for(k=0; k<nclusters; k++){
				icluster= centroids[k]+ij;
				if (dysma[i] > distmatrix[icluster]) {
					dysmb[i] = dysma[i];
					dysma[i] = distmatrix[icluster];
					tclusterid[i]=k;
				} else if (dysmb[i] > distmatrix[icluster]) {
					dysmb[i] = distmatrix[icluster];
				}
			}
		}
		//First loop, compute total sum of distances
		if(total < 0 ){
			total=0;
			for (i = 0; i < nelements; i++) {
				total+= weights[i]*dysma[i];
			}
			TMRLOG(2," Total: %g\n", total);
		}

		dzsky = 1.; /* 1 is arbitrary > 0; only dzsky < 0 matters in the end */
		TMRLOG(1, "Computing change costs\n");
		for (h = 0; h < nelements; h++) {
			//Figure out how to check that, use also nrepr?
			int hjbase =h*nelements;
			for(k=0; k<nclusters; k++){
				//Don't check if distances is zero
				if(distmatrix[hjbase+centroids[k]]==0){
					break;
				}
				/*if(h==centroids[k]){
					break;
				}*/
			}
			if (k==nclusters) {
				R_CheckUserInterrupt();
				//For each centroid
				for(k=0; k<nclusters; k++){
					i=centroids[k];
					int ijbase =i*nelements;
					double dz = 0.;
						/* dz := T_{ih} := sum_j C_{jih}  [p.104] : */
					
					for (j = 0; j < nelements; j++) { /* if (!nrepr[j]) { */
						int hj = hjbase+j;
						ij = ijbase+j;
						if (distmatrix[ij] == dysma[j]) {
							double small = dysmb[j] > distmatrix[hj]? distmatrix[hj] : dysmb[j];
							dz += weights[j]*(- dysma[j] + small);
						} else if (distmatrix[hj] < dysma[j]) /* 1c. */{
							dz += weights[j]*(- dysma[j] + distmatrix[hj]);
						}
					}
					if (dzsky > dz) {
						dzsky = dz; /* dzsky := min_{i,h} T_{i,h} */
						hbest = h;
						nbest = i;
					}
				}
			}
		}
		//REprintf( "   Current best swap %d <-> %d old; decreasing diss. by %g\n",
			//	hbest, nbest, dzsky);
		if (dzsky < 0.) { /* found an improving swap */
			TMRLOG(2, "   swp new %d <-> %d old; decreasing diss. by %g\n",
					hbest, nbest, dzsky);
				/*nrepr[hbest] = 1;
				nrepr[nbest] = 0;*/
			for(k=0; k<nclusters; k++) {
				if(centroids[k]==nbest){
					centroids[k]=hbest;
				}
			}
			/*for (icluster = 0; icluster < nclusters; icluster++){
				REprintf("%d, ", centroids[icluster]);
			}*/
			total += dzsky;
			TMRLOG(2," Total: %g\n", total);
		}
	} while(dzsky < 0.);
	TMRLOG(1,"Centroids lists (out):");
	for (icluster = 0; icluster < nclusters; icluster++){
		TMRLOG(1,"%d, \n", centroids[icluster]);
	}
	return total;
 }
 
 
