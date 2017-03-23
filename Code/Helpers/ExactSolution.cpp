
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP

#include "ExactSolution.h"


double ExactSolution::value (const Point<3> &p , const unsigned int component) const
{
	return ModeMan.get_input_component( component, p, 0);
	//return 0.0;
}


void ExactSolution::vector_value (const Point<3> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values[c] = ModeMan.get_input_component( c, p, 0);
	//std::cout << "( " << p[0] << " , " << p[1] << " , " << p[2] << ")" << std::endl;
}

ExactSolution::ExactSolution(): Function<3>(6) {

}

#endif
