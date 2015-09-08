#include "ExactSolution.h"

template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p , const unsigned int component) const
{
	return Solution(p, component);
}

template <int dim>
void ExactSolution<dim>::vector_value (const Point<dim> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values[c] = ExactSolution<dim>::value (p, c);
}

template <int dim>
ExactSolution<dim>::ExactSolution(): Function<dim>(6) {
	prm = GetParameters();
}

