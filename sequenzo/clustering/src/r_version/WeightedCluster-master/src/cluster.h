#ifndef TRAMINER_H_INCLUDED
#define TRAMINER_H_INCLUDED

//Includes usually used
/*
#ifdef __SUNPRO_CC
	#ifdef length
		#undef length
	#endif
#endif
*/
//#define R_NO_REMAP
#ifdef __cplusplus
	#include <cstdio>
	#include <cstdarg>
	using std::sprintf;
#endif	//#ifdef __cplusplus
#include <map>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>



//Macros for matrix indices
//Using C indices
#define MINDICE(ligne, colone,len) ((ligne)+(colone)*(len))

//Using R indices
#define MINDICER(ligne, colone,len) ((ligne-1)+(colone-1)*(len))


//MACROS for array of three dimension

#define ARINDICE(ligne, colone, last, len) ((ligne)+(colone)*(len)+(len)*(len)*(last))

#define TMRDISTINDEX(i,j,n) ((n)*((i)-1) - ((i)*((i)-1))/2 + (j)-(i)-1)


//MACRO FOR DIST INDICES LOOPS
//First loop
#define DL_FIRST_INIT(firstit, n) (-(n)+(firstit))
#define DL_FIRST_INC(firstit, secondit, n) ((n)-(secondit)-1)
#define DL_FIRST_ACCESS(ij, firstit, secondit) ((ij))
//Warning do not forget equality case!!!

//Second loop
#define DL_SEC_INIT(firstit, n) (((firstit)*(n))-(((firstit)+1)*(firstit))/2-(firstit)-1)
#define DL_SEC_ACCESS(ij, firstit, secondit) ((ij)+(secondit))



#ifdef TRAMINER_DEBUG_LEVEL_0
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 0
#elif defined(TRAMINER_DEBUG_LEVEL_1)
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 1
#elif defined(TRAMINER_DEBUG_LEVEL_2)
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 2
#elif defined(TRAMINER_DEBUG_LEVEL_3)
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 3
#elif defined(TRAMINER_DEBUG_LEVEL_4)
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 4
#elif defined(TRAMINER_DEBUG_LEVEL_5)
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 5
#else 
	#define TRAMINER_DEBUG_LEVEL_DEFAULT 0
#endif

extern int TRAMINER_DEBUG_LEVEL;

#ifdef TRAMINER_DEBUG
	#define TMRLOG(level,...) do{\
			if(level<TRAMINER_DEBUG_LEVEL){\
				REprintf("%s %d: ", __FILE__, __LINE__);\
				REprintf(__VA_ARGS__);\
				REprintf("\n");\
			}\
		}while(false)

	#define TMRASSERT(exp) do{\
		if(!(exp)){\
			error("TMRASSERT(%s) failed in %s %d", #exp, __FILE__, __LINE__);\
		}\
		}while(false)
	#define TMRLOGMATRIX(level,  mat, row, col, matsize) do{\
			if(level<TRAMINER_DEBUG_LEVEL){\
				REprintf("%s %d: ", __FILE__, __LINE__);\
				REprintf("Row: %d, Col:%d\n", row, col);\
				for(int i=0; i<row; i++){\
					for(int j=0; j<col; j++){\
						REprintf("%g\t", mat[MINDICE(i,j,matsize)]);\
					}\
					REprintf("\n");\
				}\
			}\
		}while(false)
#else
	#define TMRLOG(level,...) do{}while(false)
	#define TMRASSERT(exp) do{}while(false)
	#define TMRLOGMATRIX(level,  mat, row, col, matsize) do{}while(false)
#endif

// #include <time.h>
// class CodeTimer{
	// clock_t start;
	// public:
		// CodeTimer(){
			// start = clock();
		// }
		// ~CodeTimer(){}
		// void reinit(){start = clock();}
		// void stop(int level=2){
			// clock_t end = clock();
			// double cpuTime= difftime(end, start)/ ((double)CLOCKS_PER_SEC);
			// TMRLOG(level, "%g secs\n", cpuTime);
			// reinit();
		// }
// };

#endif // TRAMINER_H_INCLUDED
