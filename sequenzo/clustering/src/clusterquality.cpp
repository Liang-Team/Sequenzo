//#include <map>
#include "cluster.h"
/*R_INLINE int getSizeIndex(int* tablesizes, int clust, int maxsize){
	int i=0;
	while(i<maxsize){
		if(tablesizes[i]==clust){
			return i;
		} else if(tablesizes[i]<0){
			tablesizes[i] = clust;
			return i;
		}
		i++;
	}
	error("Cluster not found in table\n");
	return i;
	
}*/




class CmpCluster{
	public:
	double clustDist0;
	double clustDist1;
	CmpCluster():clustDist0(0), clustDist1(0){}
	~CmpCluster(){}
	
};
#define ClusterQualHPG 0
#define ClusterQualHG 1
#define ClusterQualHGSD 2
#define ClusterQualASWi 3
#define ClusterQualASWw 4
#define ClusterQualF 5
#define ClusterQualR 6
#define ClusterQualF2 7
#define ClusterQualR2 8
#define ClusterQualHC 9
#define ClusterQualNumStat 10


typedef std::map<double, CmpCluster *> KendallTree;
typedef std::map<double, CmpCluster *>::iterator KendallTreeIterator;

void finalizeKendall(SEXP ptr){
	KendallTree * kendall;
	//REprintf("Finalizing kendall\n");
	kendall= static_cast<KendallTree *>(R_ExternalPtrAddr(ptr));
	KendallTreeIterator it;
	for (it = kendall->begin();it != kendall->end();it++) {
		delete it->second;
	}
	delete kendall;
}

SEXP kendallFactory(KendallTree *kendall) {
    SEXP SDO, classname;
	PROTECT(classname = Rf_allocVector(STRSXP, 1));
	SET_STRING_ELT(classname, 0, Rf_mkChar("KendallTree"));
    SDO = R_MakeExternalPtr(kendall, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(SDO, (R_CFinalizer_t) finalizeKendall, TRUE);
    Rf_classgets(SDO, classname);
	UNPROTECT(1);
    return SDO;
}


void resetKendallTree(KendallTree * kendall){
	TMRLOG(2, "Resetting kendall\n");
	KendallTreeIterator it;
	for (it = kendall->begin();it != kendall->end();it++) {
		it->second->clustDist0=0;
		it->second->clustDist1=0;
	}
}


#define CLUSTERQUALITY_INCLUDED
	//Including version based on dist object
	#define DISTOBJECT_VERSION
	#define CLUSTERQUALITY_FUNCNAME clusterquality_dist
	#define INDIV_ASW_FUNCNAME indiv_asw_dist
	#define CLUSTERQUALITYSIMPLE_FUNCNAME clusterqualitySimple_dist
		#include "clusterqualitybody.cpp"
	#undef DISTOBJECT_VERSION
	#undef CLUSTERQUALITY_FUNCNAME 
	#undef INDIV_ASW_FUNCNAME 
	#undef CLUSTERQUALITYSIMPLE_FUNCNAME

	//Including version based on a distance matrix
	#define DISTMATRIX_VERSION
	#define CLUSTERQUALITY_FUNCNAME clusterquality
	#define INDIV_ASW_FUNCNAME indiv_asw
	#define CLUSTERQUALITYSIMPLE_FUNCNAME clusterqualitySimple
		#include "clusterqualitybody.cpp"
	#undef DISTMATRIX_VERSION
	#undef CLUSTERQUALITY_FUNCNAME 
	#undef INDIV_ASW_FUNCNAME 
	#undef CLUSTERQUALITYSIMPLE_FUNCNAME

#undef CLUSTERQUALITY_INCLUDED

extern "C" {
	SEXP RClusterQual(SEXP diss, SEXP cluster, SEXP weightSS, SEXP numclust, SEXP isdist){
		int nclusters=INTEGER(numclust)[0];
		SEXP ans, stats, asw;
		PROTECT(ans = Rf_allocVector(VECSXP, 2));
		PROTECT(stats = Rf_allocVector(REALSXP, ClusterQualNumStat));
		PROTECT(asw = Rf_allocVector(REALSXP, 2*nclusters));
		SET_VECTOR_ELT(ans, 0, stats);
		SET_VECTOR_ELT(ans, 1, asw);
		KendallTree kendall;
		if(INTEGER(isdist)[0]){
			clusterquality_dist(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), kendall);
		} else {
			clusterquality(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), kendall);
		}
		KendallTreeIterator it;
		for (it = kendall.begin();it != kendall.end();it++) {
			delete it->second;
		}
		UNPROTECT(3);
		return ans;
		
	}
	SEXP RClusterQualKendallFactory(void){
		//REprintf("Kendall factory\n");
		KendallTree *kt = new KendallTree();
		//REprintf("Kendall factory %d\n", kt->size());
		return(kendallFactory(kt));
	}
	SEXP RClusterQualKendall(SEXP diss, SEXP cluster, SEXP weightSS, SEXP numclust, SEXP isdist, SEXP kendallS){
		int nclusters=INTEGER(numclust)[0];
		SEXP ans, stats, asw;
		PROTECT(ans = Rf_allocVector(VECSXP, 2));
		PROTECT(stats = Rf_allocVector(REALSXP, ClusterQualNumStat));
		PROTECT(asw = Rf_allocVector(REALSXP, 2*nclusters));
		SET_VECTOR_ELT(ans, 0, stats);
		SET_VECTOR_ELT(ans, 1, asw);
		//REprintf("Before Kendall\n");
		KendallTree * kendall= static_cast<KendallTree *>(R_ExternalPtrAddr(kendallS));
		//REprintf("Before size\n");
		//REprintf("Kendall initialized %d\n", kendall->size());
		resetKendallTree(kendall);
		
		//REprintf("Kendall initialized\n");
		//return(ans);
		if(INTEGER(isdist)[0]){
			clusterquality_dist(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), (*kendall));
		} else {
			clusterquality(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), (*kendall));
		}
		UNPROTECT(3);
		return ans;
	}
	
	SEXP RClusterComputeIndivASW(SEXP diss, SEXP cluster, SEXP weightSS, SEXP numclust, SEXP isdist){
		int nclusters=Rf_asInteger(numclust);
		SEXP ans, asw_i, asw_w;
		PROTECT(asw_i = Rf_allocVector(REALSXP, Rf_length(cluster)));
		PROTECT(asw_w = Rf_allocVector(REALSXP, Rf_length(cluster)));
		PROTECT(ans = Rf_allocVector(VECSXP, 2));
		SET_VECTOR_ELT(ans, 0, asw_i);
		SET_VECTOR_ELT(ans, 1, asw_w);
		if(INTEGER(isdist)[0]){
			indiv_asw_dist(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), nclusters, REAL(asw_i), REAL(asw_w));
		} else {
			indiv_asw(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), nclusters, REAL(asw_i), REAL(asw_w));
		}
		UNPROTECT(3);
		return ans;
		
	}
	SEXP RClusterQualSimple(SEXP diss, SEXP cluster, SEXP weightSS, SEXP numclust, SEXP isdist){
		int nclusters=INTEGER(numclust)[0];
		SEXP stats, asw;
		PROTECT(stats = Rf_allocVector(REALSXP, ClusterQualNumStat));
		PROTECT(asw = Rf_allocVector(REALSXP, nclusters));
		if(INTEGER(isdist)[0]){
			clusterqualitySimple_dist(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw));
		} else {
			clusterqualitySimple(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw));
		}
		UNPROTECT(2);
		return stats;
		
	}
	
	//.Call(, ans, diss, as.integer(clustmat), as.double(weights), as.integer(ncluster), as.integer(R),  quote(internalsample()), environment(), samplesize, isdist, simple)
	SEXP RClusterQualBootSeveral(SEXP ans, SEXP diss, SEXP clustmatS, SEXP weightSS, SEXP numclust, SEXP Rs, SEXP expr, SEXP rho, SEXP samplesizeS, SEXP isdist, SEXP simpleS){
	
		int nbclusttest = Rf_ncols(clustmatS);
		int ncase = Rf_nrows(clustmatS);
		int * clustmat=INTEGER(clustmatS);
		//REprintf("Clustmat size =%d x %d\n", ncase, nbclusttest);
		int R = Rf_asInteger(Rs);
		bool simple = Rf_asLogical(simpleS);
		int full_stat_indice[] = {ClusterQualHPG, ClusterQualHG, ClusterQualHGSD, 
						ClusterQualASWi, ClusterQualASWw, ClusterQualF, 
						ClusterQualR, ClusterQualF2, ClusterQualR2, ClusterQualHC};
		int short_stat_indice[] = {ClusterQualHPG, ClusterQualF, ClusterQualR, ClusterQualF2, ClusterQualR2};
		int num_st_indice = ClusterQualNumStat;
		int * stat_indice = full_stat_indice;
		int samplesize=Rf_asInteger(samplesizeS);
		if(simple){
			stat_indice = short_stat_indice;
			num_st_indice = 5;
		}
		double * weights = new double[ncase];
		double * ww=NULL;
		double * stat=new double[ClusterQualNumStat];
		int maxncluster=-1;
		for(int c=0; c<nbclusttest; c++){
			int nclusters=INTEGER(numclust)[c];
			if(nclusters>maxncluster){
				maxncluster=nclusters;
			}
		}
		//REprintf("Maxncluster=%d\n", maxncluster);
		double *asw= new double[2*maxncluster];
		SEXP randomSample;
		KendallTree kendall;
		for(int r=0; r<R; r++){
			//REprintf("R=%d\n", r);
			if(r==0){
				ww = REAL(weightSS);
			}else{
				for(int i=0; i<ncase; i++){	
					weights[i]=0;
				}
				PROTECT(randomSample = Rf_eval(expr, rho));
				int* rs=INTEGER(randomSample);
				for(int i=0; i<samplesize; i++){	
					weights[rs[i]]++;
				}
				UNPROTECT(1);
				ww = weights;
			}
			for(int c=0; c<nbclusttest; c++){
				//REprintf("Starting C=%d loop\n", c);
				int nclusters=INTEGER(numclust)[c];
				int* clustsol =clustmat + c*ncase;
				if(INTEGER(isdist)[0]){
					if(simple){
						clusterqualitySimple_dist(REAL(diss), clustsol, ww, ncase, stat, nclusters, asw);
					}else{
						resetKendallTree(&kendall);
						clusterquality_dist(REAL(diss), clustsol, ww, ncase, stat, nclusters, asw, kendall);
					}
				} else {
					if(simple){
						clusterqualitySimple(REAL(diss), clustsol, ww, ncase, stat, nclusters, asw);
					}else{
						resetKendallTree(&kendall);
						clusterquality(REAL(diss), clustsol, ww, ncase, stat, nclusters, asw, kendall); 
					}
				}
				//REprintf("Copying values");
				double * stt=REAL(VECTOR_ELT(ans, c));
				for(int i=0; i<num_st_indice;i++){
					stt[r+i*R] = stat[stat_indice[i]];
					//REprintf(" [i=%d => %d, v=%g] ", i, r+i*R, stat[stat_indice[i]]);
				}
				//REprintf("Finished Copying\n");
			}
		}
		KendallTreeIterator it;
		for (it = kendall.begin();it != kendall.end();it++) {
			delete it->second;
		}
		delete [] weights;
		delete [] stat;
		delete [] asw;
		return R_NilValue;
	}
	SEXP RClusterQualInitBoot(){
		return(kendallFactory(new KendallTree()));
	}
	SEXP RClusterQualBoot(SEXP diss, SEXP cluster, SEXP weightSS, SEXP numclust, SEXP kendallS, SEXP isdist){
		int nclusters=INTEGER(numclust)[0];
		SEXP ans, stats, asw;
		PROTECT(ans = Rf_allocVector(VECSXP, 2));
		PROTECT(stats = Rf_allocVector(REALSXP, ClusterQualNumStat));
		PROTECT(asw = Rf_allocVector(REALSXP, 2*nclusters));
		SET_VECTOR_ELT(ans, 0, stats);
		SET_VECTOR_ELT(ans, 1, asw);
		KendallTree * kendall;
		kendall= static_cast<KendallTree *>(R_ExternalPtrAddr(kendallS));
		resetKendallTree(kendall);
		if(INTEGER(isdist)[0]){
			clusterquality_dist(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), (*kendall));
		} else {
			clusterquality(REAL(diss), INTEGER(cluster), REAL(weightSS), Rf_length(cluster), REAL(stats), nclusters, REAL(asw), (*kendall));
		}
		UNPROTECT(3);
		return ans;
		
	}
	
}
