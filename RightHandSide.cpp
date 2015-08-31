#include "RightHandSide.h"

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p , const unsigned int component) const
{
	if(System_Coordinate_in_Waveguide(p)){
		if(p[2] < 0) {
			if(component < 2) {
				double d2 = (Distance2D(p)/2.0)/(prm.PRM_M_C_RadiusIn + prm.PRM_M_C_RadiusOut);
				return exp(-d2*d2);

			}
		}
	}
	return 0.0;
}

template <int dim>
void RightHandSide<dim>::vector_value (const Point<dim> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values[c] = RightHandSide<dim>::value (p, c);
}

template <int dim>
RightHandSide<dim>::RightHandSide(): Function<dim>(6) {
	prm = GetParameters();
}
