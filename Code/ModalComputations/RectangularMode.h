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
#include <array>
#include <memory>
#include <vector>
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
  /** @brief Propagation constant. Not used. */
  double beta;
  /** @brief Number of dofs of the eigenvalue problem. It is used for the sizes of matrices and vectors but never assigned. */
  unsigned int n_dofs_total;
  /** @brief Number of eigenpairs that solve() provides storage for. */
  unsigned int n_eigenfunctions = 1;
  
  /** @brief Storage for the eigenvalues, resized in solve(). */
  std::vector<ComplexNumber> eigenvalues;
  /** @brief Storage for the eigenfunctions (dof vectors), resized in solve(). They are not computed because the call of the eigensolver is commented out. */
  std::vector<PETScWrappers::MPI::Vector>  eigenfunctions;
  /** @brief Not used. */
  std:: vector<DofNumber> surface_first_dofs;
  /** @brief Hardy space infinite element surfaces for the four lateral boundaries. assemble_system() uses them, but they are never created. */
  std::array<std::shared_ptr<HSIESurface>,4> surfaces;
  /** @brief The Nedelec element of order GlobalParams.Nedelec_element_order. */
  dealii::FE_NedelecSZ<3> fe;
  /** @var constraints
   *  @brief The constraints used in the assembly (empty, closed in assemble_system()). */
  /** @var periodic_constraints
   *  @brief Not used. */
  Constraints constraints, periodic_constraints;
  /** @brief Mesh of the cross section: 20 x 20 x 1 cells on [-Geometry_Size_X/2, Geometry_Size_X/2] x [-Geometry_Size_Y/2, Geometry_Size_Y/2] x [0, layer_thickness]. */
  Triangulation<3> triangulation;
  /** @brief Dof handler on triangulation. The dofs are renumbered in SortDofsDownstream(). */
  DoFHandler<3> dof_handler;
  /** @brief Sparsity pattern of both matrices. */
  SparsityPattern sp;
  /** @var mass_matrix
   *  @brief Matrix of the form (omega^2 epsilon_r u, v). */
  /** @var stiffness_matrix
   *  @brief Matrix of the form (curl u, curl v). */
  PETScWrappers::SparseMatrix mass_matrix, stiffness_matrix;
  /** @var rhs
   *  @brief Zero vector that is only initialized. */
  /** @var solution
   *  @brief Not used. */
  NumericVectorDistributed rhs, solution;
  /** @brief Thickness of the single cell layer in the z-direction in micrometers (0.05). */
  const double layer_thickness;
  /** @brief Wavelength in micrometers (1.55). Not used; the assembly reads GlobalParams.Lambda. */
  const double lambda;
  /** @brief Creates the finite element and the (empty) mesh and dof handler. */
  RectangularMode();
  /** @brief Builds the sparsity pattern and assembles mass_matrix and stiffness_matrix. */
  void assemble_system();
  /** @brief Creates the mesh (see triangulation) and distributes the dofs. */
  void make_mesh();
  /** @brief Empty apart from log messages. */
  void make_boundary_conditions();
  /** @brief Writes the permittivity per cell and the eigenfunctions to eigenvalues.vtu in the working directory. */
  void output_solution();
  /** @brief Calls make_mesh(), SortDofsDownstream(), make_boundary_conditions(), assemble_system(), solve() and output_solution(). */
  void run();
  /** @brief Allocates the eigenvalues and eigenfunctions and normalizes the eigenfunctions. The eigensolver itself is commented out. */
  void solve();
  /**
   * @brief Relative permittivity of the straight rectangular waveguide at a position.
   * @param in_position position in micrometers; only x and y are used.
   * @return double Epsilon_R_in_waveguide inside the core (|x| < width / 2 and |y| < height / 2) and
   * Epsilon_R_outside_waveguide otherwise.
   */
  static auto compute_epsilon_for_Position(Position in_position) -> double;
  /** @brief Renumbers the dofs in the order of the centers of the lines, faces and cells they belong to (by z, then y, then x; see comparePositions()). */
  void SortDofsDownstream();
  /** @brief Legacy declaration without definition. Not used. */
  IndexSet get_dofs_for_boundary_id(types::boundary_id);
  /**
   * @brief Collects the dofs on one boundary of the mesh.
   * @param b_id boundary id in [0,5].
   * @return std::vector<InterfaceDofData> dof index, base point (center of the line or face) and order of the dofs,
   * sorted with compareDofBaseDataAndOrientation().
   */
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id(unsigned int b_id);
};
