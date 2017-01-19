#ifndef WaveguideFlag
#define WaveguideFlag


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

// Trilinos Headers
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// #include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>


#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <sstream>
#include <mpi.h>

#include "../Helpers/Parameters.h"
#include "../Helpers/ParameterReader.h"
#include "./PreconditionerSweeping.h"
#include "../Helpers/staticfunctions.h"

#include "../MeshGenerators/MeshGenerator.h"
#include "../SpaceTransformations/SpaceTransformation.h"

using namespace dealii;


static Parameters GlobalParams;
static const CylindricalManifold<3, 3> round_description (2);

/**
 * \class Waveguide
 * \brief This class encapsulates all important mechanism for solving a FEM problem. In earlier versions this also included space transformation and computation of materials. Now it only includes FEM essentials and solving the system matrix.
 *
 * Upon initialization it requires structural information about the waveguide that will be simulated. The object then continues to initialize the FEM-framework. After allocating space for all objects, the assemblation-process of the system-matrix begins. Following this step, the user-selected preconditioner and solver are used to solve the system and generate outputs.
 * This class is the core piece of the implementation.
 *
 * \author Pascal Kraft
 * \date 03.07.2016
 */
class Waveguide
{
	public:
	/**
	 * This is the constructor that should be used to initialize objects of this type.
	 *
	 * \param param This is a reference to a parsed form of the input-file.
	 * \param structure This parameter gives a reference to the structure of the real Waveguide. This is necessary since during matrix-assembly it is required to call a function which generates the transformation tensor which is purely structure-dependent.
	 */

		Waveguide (MPI_Comm in_mpi_comm, MeshGenerator * in_mg, SpaceTransformation * in_st );

    ~Waveguide ();

		/**
		 * This method as well as the rerun() method, are used by the optimization-algorithm to use and reuse the Waveguide-object. Since the system-matrix consumes a lot of memory it makes sense to reuse it, rather then creating a new one for every optimization step.
		 * All properties of the object have to be created properly for this function to work.
		 */
		void 		run ();

		/**
		 * This method as well as the run() method, are used by the optimization-algorithm to use and reuse the Waveguide-object. Since the system-matrix consumes a lot of memory it makes sense to reuse it, rather then creating a new one for every optimization step.
		 * All properties of thenon intersecting planes bent object have to be created properly for this function to work.
		 */

		void 		rerun ();

		/**
		 * The assemble_part(unsigned int in_part) function is a part of the assemble_system() functionality. It builds a part of the system-matrix. assemble_system() creates the global system matrix. After splitting the degrees of freedom into several block, this method takes one block (identified by the integer passed as an argument) and calculates all matrix-entries that reference it.
		 * @param in_part Numerical identifier of the part of the system.
		 * @date 13.11.2015
		 * @author Pascal Kraft
		 */
		void 		assemble_part ();

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
		 * To compute the output quality of the signal and it's transmition along the waveguid-axis, this function performs a comparison of the fundamental mode of a waveguide and the actual situation. For this purpose we integrate the product of the two functions over a cross-section of the waveguide in transformed coordinates. To perform this action we need to use numeric integration so the integral is decomposed into a sum over local evaluations. For this to be possible this function can be handed x,y and z coordinates and returns the according value.
		 * \param x gives the x-coordinate.
		 * \param y gives the y-coordinate.
		 * \param z gives the z-coordinate.
		 */
		std::complex<double> evaluate_for_Position(double x, double y, double z);
		/**
		 * To compute the output quality of the signal and it's transmition along the waveguid-axis, this function performs a comparison of the fundamental mode of a waveguide and the actual situation. For this purpose we integrate the product of the two functions over a cross-section of the waveguide in transformed coordinates. To perform this action we need to use numeric integration so the integral is decomposed into a sum over local evaluations. For this to be possible this function can be handed x,y and z coordinates and returns the according value.
		 * \param x gives the x-coordinate.
		 * \param y gives the y-coordinate.
		 * \param z gives the z-coordinate.
		 */
		double evaluate_for_z(double);

		/**
		 * This function has the purpose of filling the qualities array in every process with the appropriate Values from the other ones.
		 * Now it will become necessary to build an optimization-scheme ontop, which can handle this information on process one and then distribute a new shape to the others. The function will use the Waveguide-Property execute_rebuild to signal a need for recomputation.
		 */
		void 		evaluate();

		/**
		 * The storage has the following purpose: Regarding the optimization-process there are two kinds of runs. The first one, taking place with no knowledge of appropriate starting values for the degrees of freedom, and the following steps, in which the prior results can be used to estimate appropriate starting values for iterative solvers as well as the preconditioner. This function switches the behaviour in the following way: Once it is called, it stores the current solution in a run-independent variable, making it available for later runs. Also it sets a flag, indicating, that prior solutions are now available for usage in the solution process.
		 */
		void 		store();

		double *										qualities;

    /**
     * This function is currently not in use. It is supposed to create a useful input-vector for the first step of the iteration. However currently this is not used, since current cases simply use a zero-vector for the first step and previous solutions in the subsequent steps.
     *
     */
    void  estimate_solution();

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
		void 	output_results (bool details);

		/**
		 * This function is used bz the GMRE-solvers in deal. This solver uses the iteration-results to estimate the eigenvalues and this function is used via handle to use them. In this function, the eigenvalues are simply pushed into a file.
		 */
		void 	print_eigenvalues(const std::vector<std::complex<double>> &);

		/**
		 * Similar to the functio print_eigenvalues(const std::vector<std::complex<double>> &) , this function uses step-results of the GMRES-solver to make properties of the system-matrix available. In this case it is the condition number, estimated on the basis of said eigenvalues, that gets pushed to a file also.
		 */
		void	print_condition (double);

		/**
		 * This function occupies one slot of the Solver and will generate formatted output on the console and write the convergence history to a file.
		 */
		SolverControl::State check_iteration_state(const unsigned int, const double, const dealii::TrilinosWrappers::MPI::Vector &);



		/**
		 * This function fills the ConstraintMatrix-object of the Waveguide-object with all constraints needed for condensation into the szstem-matrix. It's properties are derived from the Waveguide itself and the Waveguide-Structure-object available to it, therefore there are no parameters but those members need to be prepared accordingly..
		 */
		void	MakeBoundaryConditions ();

		/**
		 * This function generates the Constraint-Matrices for the two Preconditioner Matrices.
		 */
		void	MakePreconditionerBoundaryConditions ( );

		/**
		 * This function executes refined downstream ordering of degrees of freedom.
		 */
		void 	Compute_Dof_Numbers();


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
		 * Reinit all datastorage objects.
		 */
		void 	reinit_all();

		/**
		 * Reinit only the right hand side vector.
		 */
		void 	reinit_rhs();

		/**
		 * Reinit only the PML-Matrix which is used in the construction of the Preconditioner. This should only be used if the need for space is there. Otherwise this matrix while being a temporary object, is very large.
		 */
		void	reinit_preconditioner();

		/**
		 * Reinit only the system matrix.
		 */
		void 	reinit_systemmatrix();

		/**
		 * The cell weights can be used to store any scalar information about each cell of the mesh. This reinit-function prepares the data structures for its usage.
		 */
		void    reinit_cell_weights();
        
    /**
     * In calculate cell weights an arbitrary value for each cell can be computed and then this value can be sent to the output to generate a plot of it. An example for this procedure is the computation of the norm of the material tensor to check it's validity across the mesh.
     */
		void    calculate_cell_weights ();

		/**
		 * Reinit only the solution vector.
		 */
		void 	reinit_solution();

		/**
		 * This function only initializes the storage vector. Keep in mind, that a call to this function is *not* included in reinit_all().
		 */
		void 	reinit_storage();

		/**
		 * When a run has already been completed, not all data structures need to be completely be rebuilt. They only need to be emptied. This function does just that.
		 */
		void 	reinit_for_rerun();

		/**
		 * Similar to the function reinit_for_rerun but focused on the data structures used by the preconditioner.
		 */
		void 	reinit_preconditioner_fast();

		/**
		 * While the solver runs, this function performs an action on the residual. In the most common use case this action is to print it to the console or to push it to some data stream.
		 */
		SolverControl::State residual_tracker(unsigned int Iteration, double resiudal, dealii::TrilinosWrappers::MPI::BlockVector vec);

		/**
		 * This function encapsulates a library call for 2D numeric integration over a circle with given properties. It is included that this function calls evaluate_for_Position(x,y,z)
		 */
		std::complex<double> gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc);


		// HIER BEGINNT DIE NEUE VERSION...

		MeshGenerator * mg;

		SpaceTransformation * st;

		MPI_Comm mpi_comm;

		parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;

		FESystem<3>                   fe;

		parallel::distributed::Triangulation<3>     triangulation;

		dealii::TrilinosWrappers::MPI::BlockVector                   system_rhs;

		TrilinosWrappers::BlockSparseMatrix          system_matrix;

		TrilinosWrappers::BlockSparseMatrix prec_matrix_odd, prec_matrix_even;

		const bool even;

		unsigned int rank;

		FEValuesExtractors::Vector            real, imag;

		SolverControl                       solver_control;

		ConstraintMatrix                cm, cm_prec_even, cm_prec_odd;

		DoFHandler<3>                 dof_handler;

	  std::vector<IndexSet> i_prec_even_owned_row;
	  std::vector<IndexSet> i_prec_even_owned_col;
	  std::vector<IndexSet> i_prec_even_writable;
	  std::vector<IndexSet> i_prec_odd_owned_row;
	  std::vector<IndexSet> i_prec_odd_owned_col;
	  std::vector<IndexSet> i_prec_odd_writable;
	  std::vector<IndexSet> i_sys_owned;

    std::string                   solutionpath;

    TrilinosWrappers::MPI::BlockVector										solution, EstimatedSolution, ErrorOfSolution ;
		IndexSet										locally_owned_dofs, locally_relevant_dofs, locally_active_dofs, extended_relevant_dofs;
		std::vector<IndexSet>							locally_relevant_dofs_per_subdomain;

		Vector<double>                  preconditioner_rhs;

		std::vector<IndexSet>             locally_relevant_dofs_all_processors;
    IndexSet                    UpperDofs, LowerDofs;

    int                       run_number;

    int                       condition_file_counter, eigenvalue_file_counter;
    const unsigned  int                   Layers;
    std::vector<int>                Dofs_Below_Subdomain, Block_Sizes;
    ConditionalOStream                pout;
    bool                      is_stored;
    TimerOutput                   timer;
    const int                     Sectors;



		// HIER BEGINNT DIE ALTE VERSION ...




		ConstraintMatrix 								boundary_value_constraints_imaginary;
		ConstraintMatrix 								boundary_value_constraints_real;
		ConstraintMatrix								hanging_global;

		TrilinosWrappers::MPI::BlockVector										storage;
		TrilinosWrappers::MPI::BlockVector										temp_storage;
		std::ofstream									eigenvalue_file, condition_file, result_file, iteration_file;


		std::vector<IndexSet> 					set;

		bool											execute_recomputation;
		Vector<float>                                   cell_weights;
		Vector<float>                                   cell_weights_prec_1;
		Vector<float>                                   cell_weights_prec_2;
    IndexSet                                        locally_owned_cells, sweepable;
};


#endif
