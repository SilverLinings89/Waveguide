#include "ExactSolution.h"

template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p , const unsigned int component) const
{
	if(component == 0) {
		std::complex<double> res(TEMode00(p,0), 0.0);
		std::complex<double> i (0,1);
		std::complex<double> ret = res * std::exp(-1.0*i*GlobalParams.PRM_C_omega*p[2]);
		return ret.real();
	}
	if(component == 3) {
		std::complex<double> res(TEMode00(p,0), 0.0);
		std::complex<double> i (0,1);
		std::complex<double> ret = res * std::exp(-1.0*i*GlobalParams.PRM_C_omega*p[2]);
		return ret.imag();
	}
	return 0.0;
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

