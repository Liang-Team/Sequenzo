#ifdef PAMONCEBODY_INCLUDED

#if !defined(DISTOBJECT_VERSION) && !defined(DISTMATRIX_VERSION)
	#error KMEDOIDBODY version not defined
#endif


#include "PAMonce.h"

	
#ifdef DISTMATRIX_VERSION
	double PAMonce::runclusterloop(const int & ipass){
#else
	double PAMonce::runclusterloop_dist(const int & ipass){
#endif		
 
	TMRLOG(1, "PAMonce LOOP\n");
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
		
		#ifdef DISTMATRIX_VERSION
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
		#else
		//Doing it in two phase
		//first init
		for (i = 0; i < nelements; i++) {
			dysma[i] = maxdist;
			dysmb[i] = maxdist;
		}
		
		for(k=0; k<nclusters; k++){
			icluster= centroids[k];
			ij =DL_FIRST_INIT(icluster, nelements);
			for(i=0;i <icluster;i++){
				ij += DL_FIRST_INC(icluster, i, nelements);
				if (dysma[i] > distmatrix[DL_FIRST_ACCESS(ij, icluster, i)]) {
					dysmb[i] = dysma[i];
					dysma[i] = distmatrix[DL_FIRST_ACCESS(ij, icluster, i)];
					tclusterid[i]=k;
				} else if (dysmb[i] > distmatrix[DL_FIRST_ACCESS(ij, icluster, i)]) {
					dysmb[i] = distmatrix[DL_FIRST_ACCESS(ij, icluster, i)];
				}
			}
			
			// For the centroid (i==icluster)
			dysmb[icluster] = dysma[i];
			dysma[icluster] = 0.0;
			tclusterid[icluster]=k;
			//Second loop
			ij = DL_SEC_INIT(icluster, nelements);
			for(i=icluster+1;i <nelements;i++){
				if (dysma[i] > distmatrix[DL_SEC_ACCESS(ij, icluster, i)]) {
					dysmb[i] = dysma[i];
					dysma[i] = distmatrix[DL_SEC_ACCESS(ij, icluster, i)];
					tclusterid[i]=k;
				} else if (dysmb[i] > distmatrix[DL_SEC_ACCESS(ij, icluster, i)]) {
					dysmb[i] = distmatrix[DL_SEC_ACCESS(ij, icluster, i)];
				}
			}
		}
		#endif		
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
		for(k=0; k<nclusters; k++){
			i=centroids[k];
			double removeCost =.0;
			//Compute cost for removing the medoid 
			for (j = 0; j < nelements; j++) {
				if(tclusterid[j]==k){
					removeCost+=weights[j]*(dysmb[j]-dysma[j]);
					fvect[j]=dysmb[j];
				}
				else{
					fvect[j]=dysma[j];
				}
			}
			
			#ifdef DISTMATRIX_VERSION
			int ibase=i*nelements;
			//Now check possible new medoids h
			for (h = 0; h < nelements; h++) {
				R_CheckUserInterrupt();
				if(distmatrix[ibase+h]>0){ // Avoid testing for identic object
					int hbase =h*nelements;
					double addGain = removeCost;
					//Compute gain of adding h as a medoid
					for (j = 0; j < nelements; j++) {
						if(distmatrix[hbase+j]<fvect[j]){
							addGain+=weights[j]*(distmatrix[hbase+j]-fvect[j]);
						}
					}
					if (dzsky > addGain) {
						dzsky = addGain; /* dzsky := min_{i,h} T_{i,h} */
						hbest = h;
						nbest = i;
					}
				}
			}
			#else
			//Now check possible new medoids h
			
			int ibase =DL_FIRST_INIT(i, nelements);
			for (h = 0; h < i; h++) {
				ibase += DL_FIRST_INC(i, h, nelements);
				R_CheckUserInterrupt();
				if(distmatrix[DL_FIRST_ACCESS(ibase, i, h)]>0){ // Avoid testing for identic object
					double addGain = removeCost;
					//Compute gain of adding h as a medoid
					int hbase =DL_FIRST_INIT(h, nelements);
					for (j = 0; j < h; j++) {
						hbase += DL_FIRST_INC(h, j, nelements);
						if(distmatrix[DL_FIRST_ACCESS(hbase, h, j)]<fvect[j]){
							addGain+=weights[j]*(distmatrix[DL_FIRST_ACCESS(hbase, h, j)]-fvect[j]);
						}
					}
					//Equality case
					if(0.0 < fvect[h]){
						addGain+=weights[h]*(0.0-fvect[h]);
					}
					hbase=  DL_SEC_INIT(h, nelements);
					for (j = h+1; j < nelements; j++) {
						if(distmatrix[DL_SEC_ACCESS(hbase, h, j)]<fvect[j]){
							addGain+=weights[j]*(distmatrix[DL_SEC_ACCESS(hbase, h, j)]-fvect[j]);
						}
					}
					if (dzsky > addGain) {
						dzsky = addGain; /* dzsky := min_{i,h} T_{i,h} */
						hbest = h;
						nbest = i;
					}
				}
			}
			//We can avoid the equality case (i==h)
			ibase =DL_SEC_INIT(i, nelements);
			for (h = i+1; h < nelements; h++) {
				R_CheckUserInterrupt();
				if(distmatrix[DL_SEC_ACCESS(ibase, i, h)]>0){ // Avoid testing for identic object
					double addGain = removeCost;
					//Compute gain of adding h as a medoid
					int hbase =DL_FIRST_INIT(h, nelements);
					for (j = 0; j < h; j++) {
						hbase += DL_FIRST_INC(h, j, nelements);
						if(distmatrix[DL_FIRST_ACCESS(hbase, h, j)]<fvect[j]){
							addGain+=weights[j]*(distmatrix[DL_FIRST_ACCESS(hbase, h, j)]-fvect[j]);
						}
					}
					//Equality case
					if(0.0 < fvect[h]){
						addGain+=weights[h]*(0.0-fvect[h]);
					}
					hbase=  DL_SEC_INIT(h, nelements);
					for (j = h+1; j < nelements; j++) {
						if(distmatrix[DL_SEC_ACCESS(hbase, h, j)]<fvect[j]){
							addGain+=weights[j]*(distmatrix[DL_SEC_ACCESS(hbase, h, j)]-fvect[j]);
						}
					}
					if (dzsky > addGain) {
						dzsky = addGain; /* dzsky := min_{i,h} T_{i,h} */
						hbest = h;
						nbest = i;
					}
				}
			}
			#endif
		}
		//REprintf( "   Current best swap %d <-> %d old; decreasing diss. by %g\n",
			//	hbest, nbest, dzsky);
		if (dzsky < WEIGHTED_CLUST_TOL) { /* found an improving swap */
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
	} while(dzsky < WEIGHTED_CLUST_TOL);
	TMRLOG(1,"Centroids lists (out):");
	for (icluster = 0; icluster < nclusters; icluster++){
		TMRLOG(1,"%d, \n", centroids[icluster]);
	}
	return total;
 }
 
 
#endif
