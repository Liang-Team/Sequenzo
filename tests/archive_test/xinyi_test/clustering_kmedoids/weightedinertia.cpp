#include "TraMineR.h"

extern "C" {

	SEXP tmrWeightedInertiaDist(SEXP diss, SEXP diss_size, SEXP isDist, SEXP individuals, SEXP Sweights, SEXP var) {
		bool isdist = INTEGER(isDist)[0]!=0;
		int n=INTEGER(diss_size)[0];
		int ilen=Rf_length(individuals);
		int * indiv=INTEGER(individuals);
		double *dmat=REAL(diss);
		double* weights = REAL(Sweights);
	//	Rprintf("mlen = %i \n", mlen);
		TMRLOG(5, "Sweights length = %i \n", Rf_length(Sweights));
		TMRLOG(5, "ilen = %i \n", ilen);
		int i, j, i_indiv, base_indice;
		double result=0, totweights=0, iweight;
		for (i=0;i<ilen;i++) {
			i_indiv=indiv[i];
			if(isdist) {
				//n*(i-1) - i*(i-1)/2 + j-i
				base_indice =n * (i_indiv - 1)- (i_indiv *(i_indiv-1)/2)-i_indiv + (-1);
				//base_indice = (i_indiv-1)*(n-i_indiv/2)-i_indiv-1;
			} else {
				base_indice=(-1)+(i_indiv-1)*n;
			}
			iweight =weights[i_indiv-1];
			totweights+=iweight;
			for (j=i+1;j<ilen;j++) {
				result+=weights[indiv[j]-1]*iweight*dmat[base_indice+indiv[j]];
			}
		}

		TMRLOG(5, "Sum = %f, totweights= %f\n",result, totweights);
		if (totweights>0)result/=totweights;
		TMRLOG(5, "Inertia = %f\n",result);
		if (INTEGER(var)[0]) {
			if (totweights>0)result/=totweights;
		}
		return Rf_ScalarReal(result);

	}
	
	SEXP tmrWeightedDistObject(SEXP diss, SEXP Sweights) {
		int ilen=Rf_length(Sweights);
		double * weights=REAL(Sweights);
		int n=ilen;
		SEXP ans;
		PROTECT(ans = Rf_allocVector(REALSXP, (ilen*(ilen-1)/2)));
		double *wmat=REAL(ans);
		double * dmat=REAL(diss);
		int i, j, i_indiv, base_indice;
		double iweight;
		for (i=0;i<ilen;i++) {
			i_indiv=i+1;
				//n*(i-1) - i*(i-1)/2 + j-i
			base_indice =n * (i_indiv - 1)- (i_indiv *(i_indiv-1)/2)-i_indiv;
			//base_indice =n * (i)- (i_indiv *(i)/2)-i_indiv + (-1);
			//base_indice = (i_indiv-1)*(n-i_indiv/2)-i_indiv-1;
			iweight =weights[i];
			for (j=i+1;j<ilen;j++) {
				wmat[base_indice+j]=weights[j]*iweight*dmat[base_indice+j];
			}
		}
		UNPROTECT(1);
		return ans;

	}
	SEXP tmrWeightedInertiaContrib(SEXP distmatrix, SEXP individuals, SEXP Sweights) {
		int mlen=Rf_nrows(distmatrix);
		int ilen=Rf_length(individuals);
		int * indiv=INTEGER(individuals);
		double * weights=REAL(Sweights);
		SEXP ans;
		PROTECT(ans = Rf_allocVector(REALSXP, ilen));
		double *result=REAL(ans);
		double *dmat=REAL(distmatrix);
		int i_index=0;
		int i, j;
		double r;
		double totweights=0, iweight;
		for (i=0;i<ilen;i++) {
			result[i]=0;
			totweights+=weights[indiv[i]-1];
		}
		for (i=0;i<ilen;i++) {
			i_index=(-1)+(indiv[i]-1)*mlen;
			iweight =weights[indiv[i]-1];
			for (j=i+1;j<ilen;j++) {
	//      Rprintf("index coord(%i,%i)=%f\n",indiv[i],indiv[j], distmatrix[MINDICER(indiv[i],indiv[j],mlen)]);
	//      Rprintf("cindex coord(%i,%i)=%f, (%i)\n",i,j, distmatrix[MINDICER(indiv[i],indiv[j],mlen)],MINDICER(indiv[i],indiv[j],mlen));
				r=dmat[i_index+indiv[j]];
				result[i]+=r*weights[indiv[j]-1];
				result[j]+=r*iweight;
			}
			if (totweights>0)result[i]/=(double)totweights;
		}
	//  Rprintf("Sum = %f\n",(*result));
		UNPROTECT(1);
		return ans;
	//   Rprintf("Inertia = %f\n",(*result));
	}
	
	SEXP tmrWeightedInertiaContribExt(SEXP distmatrix, SEXP individuals, SEXP extindivS, SEXP Sweights) {
		int mlen=Rf_nrows(distmatrix);
		int ilen=Rf_length(individuals);
		int ilenExt=Rf_length(extindivS);
		int * indiv=INTEGER(individuals);
		int * indivExt=INTEGER(extindivS);
		double * weights=REAL(Sweights);
		int totlen=ilen+ilenExt;
		SEXP ans;
		PROTECT(ans = Rf_allocVector(REALSXP, totlen));
		double *result=REAL(ans);
		double *dmat=REAL(distmatrix);
		double totweights=0, iweight;
	//  Rprintf("mlen = %i \n", mlen);
	//  Rprintf("ilen = %i \n", ilen);
		int i_index=0;
		int r_index=0;
		int i, j;
		double r;
		//Computing group weights
		for(i=0;i<ilen;i++){
			totweights+=weights[indiv[i]-1];
		}
		for (i=0;i<totlen;i++) {
			result[i]=0;
		}
		for (i=0;i<ilen;i++) {
			i_index=(-1)+(indiv[i]-1)*mlen;
			iweight =weights[indiv[i]-1];
			for (j=i+1;j<ilen;j++) {
	//      Rprintf("index coord(%i,%i)=%f\n",indiv[i],indiv[j], distmatrix[MINDICER(indiv[i],indiv[j],mlen)]);
	//      Rprintf("cindex coord(%i,%i)=%f, (%i)\n",i,j, distmatrix[MINDICER(indiv[i],indiv[j],mlen)],MINDICER(indiv[i],indiv[j],mlen));
				r=dmat[i_index+indiv[j]];
				result[i]+=r*weights[indiv[j]-1];
				result[j]+=r*iweight;
			}
			if (totweights>0)result[i]/=(double)totweights;
		}
		int ii_index=0;
		for (i=0;i<ilenExt;i++) {
			ii_index=indivExt[i]-1;
			i_index=(-1)+(ii_index)*mlen;
			r_index=i+ilen;
			iweight =weights[ii_index];
			for (j=0;j<ilen;j++) {
				result[r_index]+=iweight*weights[indiv[j]-1]*dmat[i_index+indiv[j]];
			}
			result[r_index]/=iweight*totweights;
		}
	//  Rprintf("Sum = %f\n",(*result));
		UNPROTECT(1);
		return ans;
	//   Rprintf("Inertia = %f\n",(*result));
	}
	
	SEXP tmrWeightedInterInertia(SEXP distmatrix, SEXP grp1, SEXP grp2, SEXP Sweights) {
		int mlen=Rf_nrows(distmatrix);
		int ilen1=Rf_length(grp1);
		int ilen2=Rf_length(grp2);
		int * indiv1=INTEGER(grp1);
		int * indiv2=INTEGER(grp2);
		double *dmat=REAL(distmatrix);
		double * weights=REAL(Sweights);
		int i, j;
		double iweight=0;
		//REprintf("mlen = %i \n", mlen);
		//REprintf("ilen1 = %i \n", ilen1);
		//REprintf("ilen2 = %i \n", ilen2);

		double result=0;
		for (i=0;i<ilen1;i++) {
			iweight =weights[indiv1[i]-1];
			for (j=0;j<ilen2;j++) {
				//REprintf("index coord(%i[%f],%i[%f])=%f\n",indiv1[i],iweight,indiv2[j],weights[indiv2[j]-1], dmat[MINDICER(indiv1[i],indiv2[j],mlen)]);
				//REprintf("index coord(%i,%i)=%f\n",indiv1[i],indiv2[j], dmat[MINDICER(indiv1[i],indiv2[j],mlen)]);
	//      Rprintf("cindex coord(%i,%i)=%f, (%i)\n",i,j, distmatrix[MINDICER(indiv[i],indiv[j],mlen)],MINDICER(indiv[i],indiv[j],mlen));
				result+=iweight*weights[indiv2[j]-1]*dmat[MINDICER(indiv1[i],indiv2[j],mlen)];
			}
		}

		//Rprintf("Sum = %f\n",result);
		//if(ilen>0)result/=(double)ilen;
		//Rprintf("Inertia = %f\n",result);
		return Rf_ScalarReal(result);
	//   Rprintf("Inertia = %f\n",(*result));
	}

}
