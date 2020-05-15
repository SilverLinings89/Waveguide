// Copyright 2018 Pascal Kraft
#ifndef CODE_CORE_NUMERICPROBLEM_H_
#define CODE_CORE_NUMERICPROBLEM_H_

#include <deal.II/base/mpi.h>
#include <sys/stat.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include "../Helpers/ExactSolution.h"
#include "../Helpers/GeometryManager.h"
#include "../Helpers/ModeManager.h"
#include "../Helpers/ParameterReader.h"
#include "../Helpers/Parameters.h"
#include "../Helpers/staticfunctions.h"
#include "../MeshGenerators/SquareMeshGenerator.h"
#include "../SpaceTransformations/SpaceTransformation.h"
#include "./PreconditionerSweeping.h"
#include "../Helpers/Enums.h"
#include "../Helpers/Structs.h"
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>
#include "GlobalObjects.h"

/**
 * \class NumericProblem
 * \brief This class encapsulates all important mechanism for solving a FEM
 * problem. In earlier versions this also included space transformation and
 * computation of materials. Now it only includes FEM essentials and solving the
 * system matrix.
 *
 * Upon initialization it requires structural information about the waveguide
 * that will be simulated. The object then continues to initialize the
 * FEM-framework. After allocating space for all objects, the
 * assembly-process of the system-matrix begins. Following this step, the
 * user-selected preconditioner and solver are used to solve the system and
 * generate outputs. This class is the core piece of the implementation.
 *
 * \author Pascal Kraft
 * \date 03.07.2016
 */
class NumericProblem {
 public:
  /**
   * This is the constructor that should be used to initialize objects of this
   * type.
   *
   * \param param This is a reference to a parsed form of the input-file.
   * \param structure This parameter gives a reference to the structure of the
   * real Waveguide. This is necessary since during matrix-assembly it is
   * required to call a function which generates the transformation tensor which
   * is purely structure-dependent.
   */

  NumericProblem();

  ~NumericProblem();

  /**
   * To compute the output quality of the signal and it's transmission along the
   * waveguide-axis, this function performs a comparison of the fundamental mode
   * of a waveguide and the actual situation. For this purpose we integrate the
   * product of the two functions over a cross-section of the waveguide in
   * transformed coordinates. To perform this action we need to use numeric
   * integration so the integral is decomposed into a sum over local
   * evaluations. For this to be possible this function can be handed x,y and z
   * coordinates and returns the according value. \param x gives the
   * x-coordinate. \param y gives the y-coordinate. \param z gives the
   * z-coordinate.
   */
  double evaluate_for_z(double);

  /**
   * This function has the purpose of filling the qualities array in every
   * process with the appropriate Values from the other ones. Now it will become
   * necessary to build an optimization-scheme on top, which can handle this
   * information on process one and then distribute a new shape to the others.
   * The function will use the Waveguide-Property execute_rebuild to signal a
   * need for re-computation.
   */
  void evaluate();

  /**
   * The storage has the following purpose: Regarding the optimization-process
   * there are two kinds of runs. The first one, taking place with no knowledge
   * of appropriate starting values for the degrees of freedom, and the
   * following steps, in which the prior results can be used to estimate
   * appropriate starting values for iterative solvers as well as the
   * preconditioner. This function switches the behaviour in the following way:
   * Once it is called, it stores the current solution in a run-independent
   * variable, making it available for later runs. Also it sets a flag,
   * indicating, that prior solutions are now available for usage in the
   * solution process.
   */
  void store();

  SquareMeshGenerator *mg;

  /**
   * Grid-generation is a crucial part of any FEM-Code. This function holds all
   * functionality concerning that topic. In the current implementation we start
   * with a cubic Mesh. That mesh originally is subdivided in 5 cells per
   * dimension yielding a total of 5*5*5 = 125 cells. The central cells in the
   * x-z planes are given a cylindrical manifold-description forcing them to
   * interpolate the new points during global refinement using a circular shape
   * rather than linear interpolation. This leads to the description of a
   * cylinder included within a cube. There are currently three techniques for
   * mesh-refinement:
   * 	-# Global refinement: For such refinement-cases, any cell is subdivided
   * in the middle of any dimension. In this case every cell is split into 8 new
   * ones, increasing the number of cells massively. Pros: no hanging nodes.
   * Cons: Very many new dofs that might be in areas, where the resolution of
   * the mesh is already large enough.
   * 	-# Inner refinement: In this case, only degrees that were in the
   * original core-cells, will be refined. These are cells, which in the real
   * physical simulation are part of the waveguide-core rather then the mantle.
   * 	-# Boundary-refinement: In this case, cells are refined, that are close
   * to the boundary of the waveguide (not close to the boundary of the
   * computational domain!). To see the used definition of close, please see the
   * code.
   *
   * 	Following the creation of the mesh, the dofs are distributed to it using
   * the function setup_system(). This function only has to be used once even in
   * optimization runs since the mesh can be reused for every run. This saves a
   * lot of time especially for large cases and distributed calculations.
   */
  void make_grid();

  /**
   * In this function, the first case-specific data is used. First off we number
   * the degrees of freedom. After completion of this task we start making
   * boundary-conditions. The creation of appropriate boundary-conditions is
   * twofold:
   * 	#- Mathematical boundary conditions as described in the literature on
   * this matter. In this case we use Dirichlet boundary values that are either
   * zero-values or alternatively are calculated from the mode-distribution of
   * the incoming signal.
   * 	#- Numerical constraints from hanging nodes. The non-global refinement
   * steps cause hanging-nodes that have to be constrained to their neighbors.
   * This problem can be solved automatically by deal and uses the same
   * mechanism (constraints) as mathematical boundary values do.
   *
   * 	Constraint matrices (as constructed in this function) can be used
   * primarily in two ways. Documentation concerning this problem can be found
   * at [Constraints On Degrees Of
   * Freedom](https://www.dealii.org/developer/doxygen/deal.II/group__constraints.html).
   */
  void setup_system();

  /**
   * Assemble system is the function to build a system-matrix. This can either
   * happen incrementally or from scratch depending on if a solution has been
   * stored before or not. Essentially it splits the system in blocks and then
   * calls assemble_block(unsigned int index) for the individual blocks. This
   * function will have to be improved for incremental building of the system
   * matrix in order to proceed to upcoming versions. \author Pascal Kraft
   * \date 16.11.2015
   */
  void assemble_system();

  /**
   * This function executes refined downstream ordering of degrees of freedom.
   */
  void Compute_Dof_Numbers();

  /**
   * This function returns the transposed and complex conjugated Matrix for the
   * given Matrix. The function operates on a copy, it doesn't change the
   * arguments value. \param input This is the order 2 Tensor (Matrix) to be
   * transposed (\f$a_{ij} = a'{ji}\f$) and complex conjugated
   * (\f$\operatorname{Im}(a_{ij}) = - \operatorname{Im}(a'_{ji})\f$)
   */
  Tensor<2, 3, std::complex<double>> Conjugate_Tensor(
      Tensor<2, 3, std::complex<double>> input);

  /**
   * This function calculates the complex conjugate of every vector entry and
   * returns the result in a copy. Similar to Conjugate_Tensor(Tensor<2,3,
   * std::complex<double>> input) this function does not operate in place - it
   * operates on a copy and hence returns a new object.
   */
  Tensor<1, 3, std::complex<double>> Conjugate_Vector(
      Tensor<1, 3, std::complex<double>> input);

  /**
   * Reinitialize all data storage objects.
   */
  void reinit_all();

  /**
   * Reinit only the right hand side vector.
   */
  void reinit_rhs();

  /**
   * Reinit only the system matrix.
   */
  void reinit_systemmatrix();

  /**
   * Reinit only the solution vector.
   */
  void reinit_solution();

  /**
   * Once a solution has been computed, this function can be used to evaluate it
   * at a point position. This function is similar to the other version but it
   * doesn't return the solution but stores it in the pointer given as an
   * argument.
   */

  void solution_evaluation(Point<3, double> position, double *solution) const;

  /**
   * Same as solution_evaluation but transforms the coordinate to the dual
   * system first, so if you passed in a coordinate on the input interface, it
   * would return the solution evaluation on the output interface.
   */
  void adjoint_solution_evaluation(Point<3, double> position,
                                   double *solution) const;

  void SortDofsDownstream();

  SpaceTransformation *st;

  dealii::FESystem<3> fe;

  dealii::Triangulation<3> triangulation;

  dealii::FEValuesExtractors::Vector real;
  dealii::FEValuesExtractors::Vector imag;

  dealii::SolverControl solver_control;

  dealii::AffineConstraints<double> cm;
  dealii::SparsityPattern final_sparsity_pattern;
  unsigned int n_dofs;

  dealii::DoFHandler<3> dof_handler;

  dealii::TrilinosWrappers::SparseMatrix system_matrix;
  dealii::Vector<double> system_rhs;

  dealii::IndexSet fixed_dofs;

  std::vector<unsigned int> get_surface_dof_vector_for_boundary_id(
      unsigned int b_id);

  std::vector<unsigned int> dofs_for_cell_around_point(dealii::Point<3> &in_p);
};

#endif
