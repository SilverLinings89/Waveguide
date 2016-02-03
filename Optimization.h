#ifndef OptimizationFlag
#define OptimizationFlag

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include "Waveguide.h"
#include "Parameters.h"

using namespace dealii;

/**
 * The Optimization-class gives one of the three main parts to the project. The other two being the WaveguideStructure-class (holding all the structural information and functionality such as calculating material-tensors or changing the shape) and the Waveguide-class (offering the complete solution-process for a fixed shape). This class however determines the changes between optimization-steps and administrates the optimization-process. It stores signal-quality values for different configurations of the system and builds estimates of better shapes based upon them.
 * To fulfill its purpose it offers two functions: One is its constructor, which requires a Waveguide-object, a WaveguideStructure-object and Parameters (parsed from the input file) to already exist. The second is a run method. This function does all the heavy lifting and contains all implementation about the shape-optimization.
 * \author Pascal Kraft
 * \date 23.11.2015
 */
template< typename M, typename V >
class Optimization {
	public:
		/**
		 * This member is not to be confused with the membler n_dofs in Waveguide which stores the number of degrees of freedom in the finite element implementation. This variable however stores the number of degrees of freedom for the shape of the waveguide and is calculated by
		 * \f[ \operatorname{dofs} = (s + 1) \cdot 3 -6\f]
		 * where \f$ s \f$ is the number of sectors used to model the system. If \f$ s=1 \f$ there is no optimization, since all properties of the waveguide are predetermined by the shapes boundary-conditions.
		 */
		const int dofs;

		const int freedofs;
		/**
		 * Members like this one appear in many objects and are always used to store the parsed data from the input-file.
		 */
		const Parameters System_Parameters;

		/**
		 * This member is a handle to the Waveguide-object used in the computation. This object needs such a handle in order to be able to
		 *  -# start calculations for a certain shape
		 *  -# retrieve signal quality information after a run has been completed.
		 *
		 */
		Waveguide<M, V > &waveguide;



		/**
		 * This is a constructor for the Optimization-object. It requires the handles to the two objects it has to control and an additional structure containing the data from the input-file.
		 */
		Optimization( Parameters , Waveguide<M, V >  & );

		/**
		 * This function is the core implementation of an optimization algorithm. Currently it is very fundamental in its technical prowess which can be improved upon in later versions. Essentially, it calculates the signal quality for a configurations and for small steps in every one of the dofs. After that, the optimization-step is estimated based on difference-quotients. Following this step, a large step is computed based upon the approximation of the gradient of the signal-quality functional and the iteration starts anew. If a decrease in quality is detected, the optimization-step is undone and the step-width is reduced.
		 * This function controls both the Waveguide- and the Waveguide-structure object.
		 */
		void run();

};

#endif
