#ifndef WaveguideFlag
#define WaveguideFlag

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_matrix_base.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>

// Trilinos Headers
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// PETSc Headers
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <mpi/mpi.h>
#include <sstream>

#include "Parameters.h"
#include "ParameterReader.h"
#include "WaveguideStructure.h"
#include "FileLogger.h"
#include "FileLoggerData.h"

using namespace dealii;

static Parameters GlobalParams;

/**
 * \class Waveguide
 * This class encapsulates all important data about the abstracted shape of the Waveguide, meaning a cylinder.
 * Upon initialization it requires structural information about the waveguide that will be simulated. The object then continues to initialize the FEM-framework. After allocating space for all objects, the assemblation-process of the system-matrix begins. Following this step, the user-selected preconditioner and solver are used to solve the system and generate outputs.
 * This class is the core piece of the implementation.
 *
 * \author Pascal Kraft
 * \date 16.11.2015
 */
template <typename MatrixType, typename VectorType>
class Waveguide
{
	public:
		Waveguide (Parameters &, WaveguideStructure &);

		/**
		 * This method as well as the rerun() method, are used by the optimization-algorithm to use and reuse the Waveguide-object. Since the system-matrix consumes a lot of memory it makes sense to reuse it, rather then creating a new one for every optimization step.
		 * All properties of the object have to be created properly for this function to work.
		 */
		void 		run ();

		/**
		 * This method as well as the run() method, are used by the optimization-algorithm to use and reuse the Waveguide-object. Since the system-matrix consumes a lot of memory it makes sense to reuse it, rather then creating a new one for every optimization step.
		 * All properties of the object have to be created properly for this function to work.
		 */

		void 		rerun ();

		/**
		 * The assemble_part(unsigned int in_part) function is a part of the assemble_system() functionality. It builds a part of the system-matrix. assemble_system() creates the global system matrix. After splitting the degrees of freedom into several block, this method takes one block (identified by the integer passed as an argument) and calculates all matrix-entries that reference it.
		 * @param in_part Numerical identifier of the part of the system.
		 * @date 13.11.2015
		 * @author Pascal Kraft
		 */
		void 		assemble_part (unsigned int in_part);

		/**
		 * In order to estimate the quality of the signal transmission, the signal-intensity at the input- and output-side are required. This function along with evaluate_out() are used for that purpose. An L2-type norm is calculated to estimate the intensity of the propagating modes.
		 */
		double 		evaluate_in();

		/**
		 * In order to estimate the quality of the signal transmission, the signal-intensity at the input- and output-side are required. This function along with evaluate_in() are used for that purpose. An L2-type norm is calculated to estimate the intensity of the propagating modes.
		 */

		double 		evaluate_out();

		/**
		 * This function calls both evaluate_in() and evalutat_out(). It uses the return-values to generate an estimate for the signal-quality of the simulated system. This function should only be called, once both assemblation and solving of the system matrix are complete.
		 */
		double 		evaluate_overall();

		/**
		 * The storage has the following purpose: Regarding the optimization-process there are two kinds of runs. The first one, taking place with no knowledge of appropriate starting values for the degrees of freedom, and the following steps, in which the prior results can be used to estimate appropriate starting values for iterative solvers as well as the preconditioner. This function switches the behaviour in the following way: Once it is called, it stores the current solution in a run-independent variable, making it available for later runs. Also it sets a flag, indicating, that prior solutions are now available for usage in the solution process.
		 */
		void 		store();

		/**
		 * Calcualtion of \f$\epsilon\f$ and \f$\mu\f$ are very similar, which is why they are done in the same function. These tensors model all properties of the system:
		 * 	-# The PML-method near the boundaries,
		 * 	-# the tensor-valued material-properties due to the space transformation and
		 * 	-# the real material properties of the fibre
		 *
		 * in the specified location.
		 * \param point This parameter is used to pass the location to calculate the tensor for.
		 * \param inverse If this parameter is set to true, instead of the material tensor, its inverse will be returned. For Maxwell"s equations this makes sense, if the following parameter is set to the inverse value since \f$\epsilon\f$ and \f$\mu^{-1}\f$ are needed to assemble the system.
		 * \param epsilon: Specifies to either
		 * - calculate \f$\epsilon\f$ if true or
		 * - calcualte \f$\mu\f$ if false.
		 */
		Tensor<2,3, std::complex<double>> get_Tensor(Point<3> & point, bool inverse, bool epsilon);


	private:
		/**
		 * Grid-generation is a crucial part of any FEM-Code. This function holds all functionality concerning that topic. In the current implementation we start with a cubic Mesh. That mesh originally is subdivided in 5 cells per dimension yielding a total of 5*5*5 = 125 cells. The central cells in the x-z planes are given a cylindrical manifold-description forcing them to interpolate the new points during global refinement using a circular shape rather than linear interpolation. This leads to the description of a cylinder included within a cube. There are currently three techniques for mesh-refinement:
		 * 	-# Global refinement: For such refinement-cases, any cell is subdivided in the middle of any dimension. In this case every cell is split into 8 new ones, increasing the number of cells massively. Pros: no hanging nodes. Cons: Very many new dofs that might be in areas, where the resolution of the mesh is already large enough.
		 * 	-# Inner refinement: In this case, only degrees that were in the original core-cells, will be refined. These are cells, which in the real physical simulation are part of the waveguide-core rather then the mantle.
		 * 	-# Boundary-refinement: In this case, cells are refined, that are close to the boundary of the waveguide (not close to the boundary of the computational domain!). To see the used definition of close, please see the code.
		 *
		 * 	Following the creation of the mesh, the dofs are distributed to it using the function setup_system(). This function only has to be used once even in optimization runs since the mesh can be reused for every run. This saves a lot of time especially for large cases and distributed calculations.
		 */
		void 	make_grid ();

		/**
		 * In this function, the first case-specific data is used. First off we number the degrees of freedom. After completion of this task we start making boundary-conditions. The creation of appropriate boundary-conditions is twofold:
		 * 	#- Mathematical boundary conditions as described in the literature on this matter. In this case we use Dirichlet boundary values that are either zero-values or alternatively are calculated from the mode-distribution of the incoming signal.
		 * 	#- Numerical constraints from hanging nodes. The non-global refinement steps cause hanging-nodes that have to be constrained to their neighbors. This problem can be solved automatically by deal and uses the same mechanism (constraints) as mathematical boundary values do.
		 *
		 * 	Constraint Matrixes (as constructed in this function) can be used primarily in two ways. Documentation concerning this problem can be found at [Constraints On Degrees Of Freedom](https://www.dealii.org/developer/doxygen/deal.II/group__constraints.html).
		 */
		void 	setup_system ();

		/**
		 * Assemble system is the function to build a system-matrix. This can either happen incrementally or from scratch depending on if a solution has been stored before or not. Essentially it splits the system in blocks and then calls assemble_block(unsigned int index) for the individual blocks. This function will have to be improved for incremental building of the system matrix in order to proceed to upcoming versions.
		 * \author Pascal Kraft
		 * \date 16.11.2015
		 */
		void 	assemble_system ();

		/**
		 * This function is currently not in use. It is supposed to create a useful input-vector for the first step of the iteration. However currently this is not used, since current cases simply use a zero-vector for the first step and previous solutions in the subsequent steps.
		 *
		 */
		void	estimate_solution();

		/**
		 * Upon successful assembly of the system-matrix, the solution has to be calculated. This is done in this function. There are multiple Templates of this function for enabling switching between libraries. The Dealii implementation uses deal's native solvers as well as data-types. The other templated editions use the PETSc and Trilinos equivalents. The type of solver to be used and its parameters are specified via the parameter GYU
		 */
		void 	solve ();

		/**
		 * In case no differential implementation is used (this means, that in every step of both the optimization and the calculation of the gradient, the system-matrix and all other elements are completely rebuilt) this function is used, to clear all values out of the data-objects.
		 *
		 */
		void 	reset_changes();

		/**
		 * This function takes the Waveguides solution-vector member and exports it in a .vtk-file along with the mesh-structure to make the results visible.
		 */
		void 	output_results ();

		/**
		 * This function is used bz the GMRE-solvers in deal. This solver uses the iteration-results to estimate the eigenvalues and this function is used via handle to use them. In this function, the eigenvalues are simply pushed into a file.
		 */
		void 	print_eigenvalues(const std::vector<std::complex<double>> &);

		/**
		 * Similar to the functio print_eigenvalues(const std::vector<std::complex<double>> &) , this function uses step-results of the GMRES-solver to make properties of the system-matrix available. In this case it is the condition number, estimated on the basis of said eigenvalues, that gets pushed to a file also.
		 */
		void	print_condition (double);

		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the x-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 */
		bool	PML_in_X(Point<3> & position);
		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the y-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 */
		bool	PML_in_Y(Point<3> & position);
		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the z-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 */
		bool	PML_in_Z(Point<3> & position);

		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 */
		double 	PML_X_Distance(Point<3> & position);
		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 */
		double 	PML_Y_Distance(Point<3> & position);
		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 */
		double 	PML_Z_Distance(Point<3> & position);

		/**
		 * This function fills the ConstraintMatrix-object of the Waveguide-object with all constraints needed for condensation into the szstem-matrix. It's properties are derived from the Waveguide itself and the Waveguide-Structure-object available to it, therefore there are no parameters but those members need to be prepared accordingly..
		 */
		void	MakeBoundaryConditions ();

		/**
		 * DEPRECATED. SCHEDULED FOR REMOVAL.
		 */
		double  RHS_value(const Point<3> &, const unsigned int component);

		/**
		 * This function returns the transposed and complex conjugated Matrix for the given Matrix. The function operates on a copy, it doesn't change the arguments value.
		 * \param input This is the order 2 Tensor (Matrix) to be transposed (\f$a_{ij} = a'{ji}\f$) and complex conjugated (\f$\operatorname{Im}(a_{ij}) = - \operatorname{Im}(a'_{ji})\f$)
		 */
		Tensor<2,3, std::complex<double>> Conjugate_Tensor(Tensor<2,3, std::complex<double>> input);

		/**
		 * This function calculates the complex conjugate of every vector entry and returns the result in a copy. Similar to Conjugate_Tensor(Tensor<2,3, std::complex<double>> input) this function does not operate in place - it operates on a copy and hence returns a new object.
		 */
		Tensor<1,3, std::complex<double>> Conjugate_Vector(Tensor<1,3, std::complex<double>> input);

		/**
		 * This project is designed to keep logs of several performance parameters. These use a custom implementation of the FileLogger class. This function initializes these loggers - meaning it generates file handles such that in functional code, data can immediately logged. This functionality should either be rewritten or included from a library since this is a standard functionality.
		 */
		void	init_loggers();

		/**
		 * This function stops the precondition-timer and starts the solver-timer. Weird implementation on the side of Deal makes am odd workaround necessary to keep code readable in this case. This function is not important for functional understanding of the code or FEM.
		 */
		void 	timerupdate();

		std::string			solutionpath;

		Triangulation<3>	triangulation, triangulation_real;
		FESystem<3>			fe;
		DoFHandler<3>	dof_handler, dof_handler_real;
		VectorType		solution;
		ConstraintMatrix 	cm;

		SparsityPattern		sparsity_pattern;
		MatrixType			system_matrix;
		Parameters			&prm;
		ConstraintMatrix 	boundary_value_constraints_imaginary;
		ConstraintMatrix 	boundary_value_constraints_real;

		int 				assembly_progress;
		VectorType			storage;
		bool				is_stored;
		VectorType			system_rhs;
		FileLoggerData 		log_data;
		FileLogger 			log_constraints, log_assemble, log_precondition, log_total, log_solver;
		WaveguideStructure 	&structure;
		int 				run_number;
		int					condition_file_counter, eigenvalue_file_counter;
		std::ofstream		eigenvalue_file, condition_file, result_file;

};


#endif
