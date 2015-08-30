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
