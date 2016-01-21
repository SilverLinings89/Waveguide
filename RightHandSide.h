#ifndef RightHandSideFlag
#define RightHandSideFlag

#include <deal.II/base/function.h>

using namespace dealii;

/**
 * This class is no longer in use, since it was initially used in the projection methods of deal to impose boundary conditions. However these dont really work for Dirichlet boundary conditions on Nedelec elements, which made it necessary to implement this functionality as a part of this project.
 * However, this class can be used as an example on how to write function-objects for the usage in the deal library.
 *
 * \author Pascal Kraft
 * \date 23.11.2015
 */
template <int dim>
class RightHandSide : public Function<dim, double>
{
	public:
		/**
		 * The constructor of this class does nothing but initialize its parent class and initialize the Parameters-object using the function GetParameters().
		 */
		RightHandSide ()  ;
		/**
		 * Similar to the other function objects used in this project, this one has two main methods:
		 * -# a value function to calculate a single component of the solution vector and
		 * -# a vector_value function, which calls value for all components and builds the resulting vector from the values, it returns.
		 *
		 * Due to this general structure, the value function takes 2 arguments:
		 * \param p This is the Point at which to evaluate the function.
		 * \param component This is the vectors component to be computed and ranges from zero to dim-1.
		 */
		virtual double value (const Point<dim> &p, const unsigned int component ) const;

		/**
		 * As described previously, this function calls value to compute the individual components. The type is void since the result is handed over via a reference.
		 * \param p The point at which the function should be evaluated.
		 * \param value The reference to the object in which to store the results.
		 */
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;

	private:
		/**
		 * This member gets initialized in the constructor using the GetParameters() global static function on the input file. This offers the possibility of using input parameters in this function-object. This offers the functionality of writing Run-time dynamic functions.
		 */
		Parameters prm;

};

#endif
