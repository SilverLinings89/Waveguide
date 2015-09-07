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
