/**
 * Die KLasse Solution-Weight
 * Bei der Berechnung der L2-Norm sollen nur Werte innerhalb des Wellenleiters betrachtet werden. Dies kann mit einer gewichtungs-Funktion eingestellt werden, die bei integrate_difference mitübergeben wird. Diese Klasse dient genau dazu 1 zurück zu geben, wenn die Koordinate im Wellenleiter und nicht in der PML liegt und 0 sonst.
 *
 * Funktionen: Wie immer bei abgeleiteten Klassen von Function.
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */


#ifndef SolutionWeightFlag
#define SolutionWeightFlag

#include <deal.II/base/function.h>

using namespace dealii;

template <int dim>
class SolutionWeight : public Function<dim, double>
{
	public:
		SolutionWeight ()  ;
		virtual double value (const Point<dim> &p, const unsigned int component ) const;
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;

	private:
		Parameters prm;

};

#endif
