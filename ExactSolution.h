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
