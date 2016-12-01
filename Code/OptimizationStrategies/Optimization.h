#ifndef OptimizationFlag
#define OptimizationFlag

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include "../Core/Waveguide.h"
#include "../Helpers/Parameters.h"

using namespace dealii;

/**
 * \class Optimization
 * \brief This class is an abstract interface to describe the general workings of an optimization scheme. It is used to compute optimization steps and controls the DOFs of the ShapeTransformation.
 *
 * In general there are two kinds of Optimization Schemes derived from this class. On the one hand there are finite-difference kind schemes which are based on the idea of varying the value of one shape parameter slightly, resolving the problem (which is now slightl varied compaired to the original one) and hence computing the entry of the shape gradient. Repeating this pattern for any un-restrained dof we can  compute the complete gradient.
 * The other version is based on an adjoint model where we solve the forward problem and its dual and can compute the shape gradient for all DOFs from these two results.
 * \author Pascal Kraft
 * \date 28.11.2016
 */

class Optimization {
	public:

    const int type = -1; // This means that this is not actually an Optimization-implementation. 0 = FD, 1 = Adj.

		ConditionalOStream	pout;

		const int dofs;

		const int freedofs;

		Waveguide waveguide;

		SpaceTransformation st;

		MeshGenerator mg;

		OptimizationAlgorithm  * oa;

		virtual Optimization( Parameters , Waveguide  & );

		virtual ~Optimization();

		/**
		 * This function is the core implementation of an optimization algorithm. Currently it is very fundamental in its technical prowess which can be improved upon in later versions. Essentially, it calculates the signal quality for a configurations and for small steps in every one of the dofs. After that, the optimization-step is estimated based on difference-quotients. Following this step, a large step is computed based upon the approximation of the gradient of the signal-quality functional and the iteration starts anew. If a decrease in quality is detected, the optimization-step is undone and the step-width is reduced.
		 * This function controls both the Waveguide- and the Waveguide-structure object.
		 */
		virtual void run() = 0;

};

#endif
