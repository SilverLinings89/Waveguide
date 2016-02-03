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

/**
 * SolutionWeight is a class, that has been derived from the class Function and which can be used to generate handles to functions, that return specific values. The pattern of passing object built from classes derived from Function is a commonly used one in Deal.II. This function offers a weight for locations inside the waveguide. In order to integrate or calculate the L2-norm inside the Waveguide, this is needed. The value of the function to be used is multiplied with this weighing-function which returns 1 for any point inside the Waveguide and 0 outside of it.
 * Mathematically, this function can be represented by \f[ f: \mathbb{R}^{dim} \to \mathbb{R}^{2 \cdot dim} , \, \boldsymbol{x} \mapsto \begin{cases}(1, \ldots, 1)^T \in \mathbb{R}^{dim}&\text{ for} \, \boldsymbol{x} \in \Omega_W \\ \boldsymbol{0} &\text{ else}\end{cases}
 *  \f], where \f$ \Omega_W\f$ is the set of all points contained in the Waveguide.
 */
template <int dim>
class SolutionWeight : public Function<dim, double>
{
	public:
	/**
	 * The whole class contains no specific data. The only information it needs stem from the Parameters object parsed from the input file. This object gets initialized inside this constructor.
	 */
		SolutionWeight ()  ;

		/**
		 * This function returns 1, if the given component is inside the Waveguide and 0 otherwise. Since this method is intended also for vector-valued functions , this method also has to account for a component of the result which for scalar functions is 0.
		 * \param p This is the location in which the test should be executed.
		 * \param component Determines which component is to be evaluated. In this case that information does not have any further meaning.
		 */
		virtual double value (const Point<dim> &p, const unsigned int component ) const;

		/**
		 * This function gets called by the framework and calls value(const Point<dim> &p, const unsigned int component ) for all components and stores the results in value.
		 * \param p The location is given here and gets passed along to the individual value-calls.
		 * \param value This is a vector which returns the results in place. It is a reference that is edited in value(const Point<dim> &p, const unsigned int component ).
		 */
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;


};

#endif
