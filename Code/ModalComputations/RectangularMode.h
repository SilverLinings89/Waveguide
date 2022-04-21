#pragma once
/**
 * @file RectangularMode.h
 * @author Pascal Kraft
 * @brief This is no longer active code
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
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
#include <deal.II/fe/fe_nedelec_sz.h>
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
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include "../Core/Types.h"
#include "../BoundaryCondition/HSIESurface.h"

using namespace dealii;

/**
 * @brief Legacy code. 
 * 
 * This object was intended to become a mode solver but numerical results have shown that an exact computation is not required. It is simpler to use provided mode profiles that are computed offline.
 * 
 */
class RectangularMode {
  public:
  double beta;
  unsigned int n_dofs_total;
  unsigned int n_eigenfunctions = 1;
  
  std::vector<ComplexNumber> eigenvalues;
  std::vector<PETScWrappers::MPI::Vector>  eigenfunctions;
  std:: vector<DofNumber> surface_first_dofs;
  std::array<std::shared_ptr<HSIESurface>,4> surfaces;
  dealii::FE_NedelecSZ<3> fe;
  Constraints constraints, periodic_constraints;
  Triangulation<3> triangulation;
  DoFHandler<3> dof_handler;
  SparsityPattern sp;
  PETScWrappers::SparseMatrix mass_matrix, stiffness_matrix;
  NumericVectorDistributed rhs, solution;
  const double layer_thickness;
  const double lambda;
  RectangularMode();
  void assemble_system();
  void make_mesh();
  void make_boundary_conditions();
  void output_solution();
  void run();
  void solve();
  static auto compute_epsilon_for_Position(Position in_position) -> double;
  void SortDofsDownstream();
  IndexSet get_dofs_for_boundary_id(types::boundary_id);
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id(unsigned int b_id);
};
