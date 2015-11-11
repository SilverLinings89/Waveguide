/**
 * Die RightHandSide-Klasse
 * Die Projektionsmethoden, die Deal zur Verfügung stellt, fordern als Argument eine von Function abgeleitete Klasse, die die Methoden value und vector_value überläd.
 * Value berechnet dabei den Wert der in component übergebenen Komponente und vector_value setzt die Komponenten zu einem Vektor zusammen.
 * Die Klasse wird in der abschließenden nicht mehr verwendet, weil die Projektionsmethoden für diese Anwendung nicht geeignet sind - sie kann aber als Beispiel dienen.
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */

#ifndef RightHandSideFlag
#define RightHandSideFlag

#include <deal.II/base/function.h>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim, double>
{
	public:
		RightHandSide ()  ;
		virtual double value (const Point<dim> &p, const unsigned int component ) const;
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;

	private:
		Parameters prm;

};

#endif
