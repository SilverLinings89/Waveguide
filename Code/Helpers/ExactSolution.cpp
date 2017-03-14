
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP

#include "ExactSolution.h"


double ExactSolution::value (const Point<3> &p , const unsigned int component) const
{
	return TEMode00(p, component);
	//return 0.0;
}


void ExactSolution::vector_value (const Point<3> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values[c] = TEMode00(p, c);
	//std::cout << "( " << p[0] << " , " << p[1] << " , " << p[2] << ")" << std::endl;
}

ExactSolution::ExactSolution(): Function<3>(6) {

}

#endif
