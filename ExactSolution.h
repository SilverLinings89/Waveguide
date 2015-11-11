/**
 * Die Klasse Exact-Solution
 * Diese Klasse ist eine Ableitung von der Function-Klasse und dient dazu für einen geraden Wellenleiter den L2-Fehler zu berechnen. Sie implementiert die analytische Lösung und kann wird in der Methode integrate_difference in Waveguide.cpp genutzt.
 * Sie hat die typischen Parameter einer von Function abgeleiteten Klasse. Wichtig ist dabei immer, dass hier dim und spacedim verschieden sind. Der Raum ist dreidimensional, die Vektoren sind 6-Dimensional. Man bekommt also Einträge in component von 0 bis 5.
 *
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef ExactSolutionFlag
#define ExactSolutionFlag

#include <deal.II/base/function.h>

using namespace dealii;

template <int dim>
class ExactSolution : public Function<dim, double>
{
	public:
		ExactSolution ()  ;
		virtual double value (const Point<dim> &p, const unsigned int component ) const;
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;

	private:
		Parameters prm;

};

#endif
