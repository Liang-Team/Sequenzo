#include "PAMonce.h"

PAMonce::PAMonce(SEXP Snelement, SEXP diss, SEXP _expr, SEXP _rho, SEXP Scentroids, SEXP Snpass, SEXP Sweights, SEXP Sisdist): 
			PAM(Snelement, diss, _expr, _rho, Scentroids, Snpass, Sweights, Sisdist){
				TMRLOG(1, "Start PAMonce builded\n");
				this->fvect= new double[nelements];
				TMRLOG(1, "PAMonce builded\n");
}


#define PAMONCEBODY_INCLUDED
	//Including version based on dist object
	#define DISTOBJECT_VERSION
		#include "PAMoncebody.cpp"
	#undef DISTOBJECT_VERSION

	//Including version based on a distance matrix
	#define DISTMATRIX_VERSION
		#include "PAMoncebody.cpp"
	#undef DISTMATRIX_VERSION

#undef PAMONCEBODY_INCLUDED

 
