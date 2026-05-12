#include "OMVIdistance.h"
OMVIdistance::OMVIdistance(SEXP normS, SEXP Ssequences, SEXP seqdim, SEXP lenS)
	:OMdistance(normS, Ssequences, seqdim, lenS){
}
OMVIdistance::OMVIdistance(OMVIdistance *dc)
	:OMdistance(dc){
	this->indelCalc = dc->indelCalc->copy();
}

void OMVIdistance::setParameters(SEXP params){
	OMdistance::setParameters(params);
	int indelmethod = INTEGER(getListElement(params, "indelmethod"))[0];
	if(indelmethod==0){
		indelCalc= new VaryingIndelCalculator(REAL(getListElement(params, "indels")));
	}else if(indelmethod==1){
		indelCalc= new OMlocIndelCalculator(REAL(getListElement(params, "timecost"))[0]*maxscost, REAL(getListElement(params, "localcost"))[0], this->scost, this->alphasize);
	} else {
		indelCalc= new OMlocIndelCalculatorMin(REAL(getListElement(params, "timecost"))[0]*maxscost, REAL(getListElement(params, "localcost"))[0], this->scost, this->alphasize);
	}
	//timecost = REAL(getListElement(params, "timecost"))[0]*maxscost;
	//localcost = REAL(getListElement(params, "localcost"))[0];
}

OMVIdistance::~OMVIdistance(){
	delete indelCalc;
}
double OMVIdistance::distance(const int&is, const int& js){
     //On passe les prefix commun
    double minimum=0, j_indel=0, sub=0;//, lenmax=0;
    //etats comparés
    double maxpossiblecost;
    int i=1;
    int j=1;
    int m=slen[is];
    int n=slen[js];
	double ml, nl;
	//int minlen = imin2(m, n);
	int mSuf = m+1, nSuf = n+1;
    int prefix=0;
	//Skipping common prefix
	/* TMRLOG(6,"Skipping common prefix\n");
    while (i<mSuf && j<nSuf && sequences[MINDICE(is,i-1,nseq)]==sequences[MINDICE(js,j-1,nseq)]) {
        i++;
        j++;
        prefix++;
    }
	//Skipping common suffix
	TMRLOG(6,"Skipping common suffix\n");
	while (mSuf>i && nSuf>j && sequences[MINDICE(is,(mSuf-2),nseq)]==sequences[MINDICE(js,(nSuf-2),nseq)]) {
		mSuf--;
		nSuf--;
    }
	TMRLOG(6,"Skipping common suffix\n"); */
	
	TMRLOG(5,"m =%d, n=%d, mSuf=%d, nSuf=%d i=%d, j=%d, prefix=%d, fmatsize=%d\n", m, n, mSuf, nSuf, i, j, prefix, fmatsize);
    //+1 pour correspondre a la matrice F
	int fmat_ij_prefix=0;
	int i_state_indice=0;
	int prev_istate=0, prev_jstate, j_state, i_state;
	int firststate= imax2(prefix-1, 0);
	fmat[0] = 0;
	prev_jstate=sequences[MINDICE(js,prefix,nseq)];
	j_state=sequences[MINDICE(js, firststate, nseq)];
	TMRLOG(5,"prev_jstate =%d, j_state=%d\n", prev_jstate, j_state);
	for(int ii=prefix+1; ii<mSuf; ii++){
		i_state=sequences[MINDICE(is, ii-1, nseq)];
		fmat[MINDICE(ii-prefix,0,fmatsize)] = fmat[MINDICE(ii-prefix-1,0,fmatsize)]+
				getIndel(sequences[MINDICE(is, ii-1, nseq)], 
						prev_jstate, 
						j_state);
	}
	TMRLOGMATRIX(10,  fmat, mSuf-prefix, nSuf-prefix, fmatsize);
	prev_istate=sequences[MINDICE(is,prefix,nseq)];
	i_state=sequences[MINDICE(is, firststate, nseq)];
	TMRLOG(5,"prev_istate =%d, i_state=%d\n", prev_istate, i_state);
	for(int ii=prefix+1; ii<nSuf; ii++){
		fmat[MINDICE(0,ii-prefix,fmatsize)] = fmat[MINDICE(0,ii-prefix-1,fmatsize)]+
				getIndel(sequences[MINDICE(js, ii-1, nseq)], 
						prev_istate, 
						i_state); 
		TMRLOG(5,"ii=%d, j_state =%d, indel=%g, current=%g, prev=%g\n", ii, sequences[MINDICE(js, ii-1, nseq)], 
					getIndel(sequences[MINDICE(js, ii-1, nseq)], prev_istate, i_state), 
					fmat[MINDICE(0,ii-prefix,fmatsize)],
					fmat[MINDICE(0,ii-prefix-1,fmatsize)]);
		//prev_istate=i_state;
	}
	
	TMRLOG(5,"Fmat initialized\n");
	TMRLOGMATRIX(10,  fmat, mSuf-prefix, nSuf-prefix, fmatsize);
	//prev_jstate=sequences[MINDICE(js, firststate, nseq)];
    while (j<nSuf) {
        i=prefix+1;
		fmat_ij_prefix=1 + ((j-prefix)*fmatsize);
		j_state=sequences[MINDICE(js,j-1,nseq)];
		//j_indel_val = getIndel(j_state, sequences[MINDICE(js, (j==1?1:(j-2)),nseq)], sequences[MINDICE(js,(j==n?(n-2):j), nseq)]);
		//next_jstate= sequences[MINDICE(js,(j==n?(n-2):j), nseq)];
		i_state_indice=is+prefix*nseq;
		prev_istate =sequences[MINDICE(is, firststate,nseq)];
        while (i<mSuf) {
            //i_state=sequences[MINDICE(is,i-1,nseq)];
			//TMRLOG(6,"Getting i state\n");
            i_state=sequences[i_state_indice];
			//////////////////////////////
            //Computing current indel cost
			//////////////////////////////
			//fmat_ij_prefix=((i-prefix)+(j-prefix)*(fmatsize));
			//minimum=fmat[MINDICE(i-prefix,j-1-prefix,fmatsize)]+ indel;
			//TMRLOG(6,"fmat_ij_prefix =%d,th =%d \n", fmat_ij_prefix, (MINDICE(i-prefix,j-prefix,fmatsize)));
			minimum=fmat[fmat_ij_prefix-1]+getIndel(i_state, prev_jstate, j_state);
			j_indel=fmat[fmat_ij_prefix-fmatsize]+getIndel(j_state, prev_istate, i_state);
            //j_indel=fmat[fmat_ij_prefix-fmatsize]+j_indel_val;
			if(minimum>j_indel){
				minimum=j_indel;
			}
		
			//////////////////////////////
            //Computing current indel cost
			//////////////////////////////
			//sub=fmat[MINDICE(i-1-prefix,j-1-prefix,fmatsize)]+ cost;
			//TMRLOG(6,"Substitution cost\n");
			if (i_state == j_state) {
                sub=fmat[fmat_ij_prefix-1-fmatsize];
            } else {
				//TMRLOG(6,"Sub cost\n");
                sub=fmat[fmat_ij_prefix-1-fmatsize]+ scost[MINDICE(i_state,j_state,alphasize)];
            }
            //sub=fmat[fmat_ij_prefix-1-fmatsize]+ cost;
            if (sub<minimum) {
				fmat[fmat_ij_prefix]=sub;
			} else {
				fmat[fmat_ij_prefix]=minimum;
			}
            //fmat[MINDICE(i-prefix,j-prefix,fmatsize)]=minimum;
            i++;
			prev_istate=i_state;
			fmat_ij_prefix++;
			i_state_indice+=nseq;
			
        }
		prev_jstate=j_state;
		j++;
    }//Fmat build
	//Max possible cost
    maxpossiblecost=abs(n-m)*indel+maxscost*fmin2((double)m,(double)n);
	
	TMRLOG(6,"End of dist compute index %d val %g\n", MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize), fmat[MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize)]);
	if(MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize)<0||MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize)>fmatsize*fmatsize){
		TMRLOG(4,"End of dist compute index %d val %g\n", MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize), fmat[MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize)]);
		TMRLOG(4,"m =%d, n=%d, mSuf=%d, nSuf=%d i=%d, j=%d, prefix=%d, fmatsize=%d\n", m, n, mSuf, nSuf, i, j, prefix, fmatsize);
		TMRLOG(4,"is =%d, js=%d\n", is, js);
	}
	TMRLOGMATRIX(10,  fmat, mSuf-prefix, nSuf-prefix, fmatsize);

	nl = double(n) * indel;
	ml = double(m) * indel;

    return normalizeDistance(fmat[MINDICE(mSuf-1-prefix,nSuf-1-prefix,fmatsize)], maxpossiblecost, ml, nl);
}
