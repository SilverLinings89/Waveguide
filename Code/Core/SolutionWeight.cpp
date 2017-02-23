
#include "SolutionWeight.h"

template <int dim>
double SolutionWeight<dim>::value (const Point<dim> &p, const unsigned int ) const
{
	double value = Distance2D(p);
	if(value < (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0){
		if(p[2] < GlobalParams.M_R_ZLength/2.0)
		{
			return 1.0;
		} else {
			return 0.0;
		}
	}
	else return 0.0;
}

template <int dim>
void SolutionWeight<dim>::vector_value (const Point<dim> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values[c] = SolutionWeight<dim>::value (p, c);
}

template <int dim>
SolutionWeight<dim>::SolutionWeight(): Function<dim>(6) {

}