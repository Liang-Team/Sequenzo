#ifndef _CONSTRAINT_INCLUDED_
#define _CONSTRAINT_INCLUDED_
#include <R.h>
#include <Rinternals.h>

class Constraint {

protected:
	double maxGap;
	double windowSize;
	double ageMinBegin;
	double ageMaxBegin;
	double ageMaxEnd;
	int countMethod;
public:
	Constraint(const double &mg, 
		   const double &ws, 
		   const double &aminb, 
		   const double &amaxb, 
		   const double &amaxe, 
		   const int &cmethod);
	virtual ~Constraint() 
	  {
	    
	  }
	inline double getmaxGap() 
	{
	  return(this->maxGap);
	}
	inline double getwindowSize() 
	{
	  return(this->windowSize);
	}
	inline double getageMinBegin() 
	{
	  return(this->ageMinBegin);
	}
	inline double getageMaxBegin()
	{
	  return(this->ageMaxBegin);
	}
	inline double getageMaxEnd() 
	{
	  return(this->ageMaxEnd);
	}
	inline int getcountMethod() 
	{
	  return(this->countMethod);
	}
};
#endif
