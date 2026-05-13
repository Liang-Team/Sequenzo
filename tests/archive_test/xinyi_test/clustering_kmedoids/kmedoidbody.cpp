/* Par of the code below is inspired by the C clustering library 
 * (But it has been completely rewritten). Here is the original copyright notice
 * The C clustering library.
 * Copyright (C) 2002 Michiel Jan Laurens de Hoon.
 *
 * This library was written at the Laboratory of DNA Information Analysis,
 * Human Genome Center, Institute of Medical Science, University of Tokyo,
 * 4-6-1 Shirokanedai, Minato-ku, Tokyo 108-8639, Japan.
 * Contact: mdehoon 'AT' gsc.riken.jp
 * 
 * Permission to use, copy, modify, and distribute this software and its
 * documentation with or without modifications and for any purpose and
 * without fee is hereby granted, provided that any copyright notices
 * appear in all copies and that both those copyright notices and this
 * permission notice appear in supporting documentation, and that the
 * names of the contributors or copyright holders not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific prior permission.
 * 
 * THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT
 * OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOFTWARE.
 * 
 */

#ifdef KMEDOIDBODY_INCLUDED

#if !defined(DISTOBJECT_VERSION) && !defined(DISTMATRIX_VERSION)
	#error KMEDOIDBODY version not defined
#endif

#include "kmedoid.h"
	


#ifdef DISTMATRIX_VERSION
	double KMedoid::runclusterloop(const int & ipass){
#else
	double KMedoid::runclusterloop_dist(const int & ipass){
#endif		
	TMRLOG(1, "KMEDOID LOOP\n");
	int i, j, icluster, ij;
	double total = DBL_MAX;
	int counter = 0;
	int period = 10;
	while(1) {
		R_CheckUserInterrupt();
		double previous = total;
		total = 0.0;
	////	REprintf("pass %i\n", counter);
		if(counter > 0){
			//REprintf("Enter getclustermedoids2\n");
			#ifdef DISTMATRIX_VERSION
			this->getclustermedoids();
			#else
			this->getclustermedoids_dist();
			#endif		

			//REprintf("Leaving getclustermedoids2\n\n");
		}
		if (counter % period == 0) {/* Save the current cluster assignments */
			for (i = 0; i < nelements; i++) {
				saved[i] = tclusterid[i];
			}
			if (period < INT_MAX / 2) {
				period *= 2;
			}
		}
		counter++;
		
		/* Find the center */
		//REprintf("Current medoids= ");
		for (icluster = 0; icluster < nclusters; icluster++){
			clusterSize[icluster]=0;
			//REprintf("%d, ", centroids[icluster]);
		}
		//REprintf("\nEntering Find the closest cluster:\n");
		#ifdef DISTMATRIX_VERSION
		for (i = 0; i < nelements; i++) { /* Find the closest cluster */
			double distance = DBL_MAX;
			ij = i*nelements;
			for (icluster = 0; icluster < nclusters; icluster++) { 
				double tdistance;
				j = centroids[icluster];
				if (i==j) { 
					distance = 0.0;
					tclusterid[i] = icluster;
					break;
				}
		
					//Using C indices
					//#define MINDICE(ligne, colone,len) ((ligne)+(colone)*(len))
				tdistance = distmatrix[j+ij];
				
				if (tdistance < distance) { 
					distance = tdistance;
					tclusterid[i] = icluster;
				}
			}
			//REprintf("%d is in %d clusterMembership pos %d (%d)\n", i, tclusterid[i], clusterSize[tclusterid[i]], MINDICE(clusterSize[tclusterid[i]], tclusterid[i], nelements));
			clusterMembership[MINDICE(clusterSize[tclusterid[i]], tclusterid[i], nelements)]=i;
			clusterSize[tclusterid[i]]++;
			total += weights[i] * distance;
		}
		#else
		for (i = 0; i < nelements; i++) { /* Find the closest cluster */
			double distance = DBL_MAX;
			ij = i*nelements-((i+1)*i)/2-i-1;
			for (icluster = 0; icluster < nclusters; icluster++) { 
				double tdistance;
				j = centroids[icluster];
				if(i<j){
					tdistance = distmatrix[j+ij];
				}else if (i==j) { 
					distance = 0.0;
					tclusterid[i] = icluster;
					break;
				}else{
					tdistance = distmatrix[j*nelements-((j+1)*j)/2-j-1+i];
				}
		
					//Using C indices
					//#define MINDICE(ligne, colone,len) ((ligne)+(colone)*(len))
				
				
				if (tdistance < distance) { 
					distance = tdistance;
					tclusterid[i] = icluster;
				}
			}
			//REprintf("%d is in %d clusterMembership pos %d (%d)\n", i, tclusterid[i], clusterSize[tclusterid[i]], MINDICE(clusterSize[tclusterid[i]], tclusterid[i], nelements));
			clusterMembership[MINDICE(clusterSize[tclusterid[i]], tclusterid[i], nelements)]=i;
			clusterSize[tclusterid[i]]++;
			total += weights[i] * distance;
		}
		
		#endif		
		/*REprintf("\n\nOK\n\n");
		REprintf("Current medoids= ");
		for (icluster = 0; icluster < nclusters; icluster++){
			//REprintf("\nMedoids %d (t=%d)\n: ", centroids[icluster],clusterSize[icluster]);
			for (i = 0; i < clusterSize[icluster]; i++) {
				REprintf("%d, ", clusterMembership[MINDICE(i, icluster, nelements)]);
			}
		}*/
		if (total>=previous) {
			return previous;
		}
      /* total>=previous is FALSE on some machines even if total and previous
       * are bitwise identical. */
		for (i = 0; i < nelements; i++) {
			if (saved[i]!=tclusterid[i]) {
				break;
			}
		}
		if (i==nelements) {
			return total; /* Identical solution found; break out of this loop */
		}
	}
	return total;
 }
 
#ifdef DISTMATRIX_VERSION
void KMedoid::getclustermedoids()
#else
void KMedoid::getclustermedoids_dist()
#endif		
{ 
	//REprintf("getclustermedoids\n");
	int i,ii, j, jj, k, ij, size;
	double bestMedoid, current;
	int bestMedID=0;
	for (k = 0; k < nclusters; k++) {
		size = clusterSize[k];
		//REprintf("Cluster %d (cent=%d) is size %d\n", k, centroids[k], size);
		bestMedoid=DBL_MAX;
		current=0;
		bestMedID=0;
		for(i=0; i <size;i++) {
			#ifdef DISTMATRIX_VERSION
			ii =clusterMembership[MINDICE(i, k, nelements)];
			ij = ii*nelements;
			current=0;
			for(j=0;j<size;j++){
				if(i==j){
					continue;
				}
				jj=clusterMembership[MINDICE(j, k, nelements)];
				current+=weights[jj] * distmatrix[ij+jj];
				if(current>=bestMedoid){
					//REprintf("Breaking loop (%f>=%f)\n", current, bestMedoid);
					break;
				}
			}
			#else
			ii =clusterMembership[MINDICE(i, k, nelements)];
			ij = ii*nelements-((ii+1)*ii)/2-ii-1;
			current=0;
			for(j=0;j<size;j++){
				if(i==j){
					continue;
				}
				jj=clusterMembership[MINDICE(j, k, nelements)];
				if(ii<jj){
					current+=weights[jj] * distmatrix[ij+jj];
				}else{
						current+=weights[jj] * distmatrix[jj*nelements-((jj+1)*jj)/2-jj-1+ii];
				}
				if(current>=bestMedoid){
					//REprintf("Breaking loop (%f>=%f)\n", current, bestMedoid);
					break;
				}
			}
			#endif
			if(current<bestMedoid){
				bestMedoid=current;
				bestMedID=ii;
				//REprintf("Breaking loop (%f>=%f)\n", current, bestMedoid);
			}
		}
		//REprintf("Best is %d with %f\n", bestMedID, bestMedoid);
//		[k]=bestMedoid;
		centroids[k]=bestMedID;
	}
}
#endif
