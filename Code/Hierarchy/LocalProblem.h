#pragma once

#include <deal.II/lac/petsc_sparse_matrix.h>
#include "HierarchicalProblem.h"
#include "../Core/InnerDomain.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "deal.II/lac/petsc_solver.h"
#include <array>
#include <memory>

class LocalProblem: public HierarchicalProblem {
public:
  SolverControl sc;
  dealii::PETScWrappers::SparseDirectMUMPS solver;
  
  /**
   * @brief Construct a new LocalProblem object
   * This initializes the local solver object and the matrix (not its sparsity pattern). It also copies the set of locally owned dofs.
   */
  LocalProblem();

  /**
   * @brief Deletes the system matrix.
   * 
   */
  ~LocalProblem() override;

  /**
   * @brief Calls the direct sovler.
   * 
   */
  void solve() override;

  /**
   * @brief Calls the reinitialization of the data structures.
   * 
   */
  void initialize() override;

  /**
   * @brief Assembles the local problem (inner domain and boundary methods).
   * 
   */
  void assemble() override;

  /**
   * @brief For local problems this is relatively simple because all locally active dofs are also locally owned.
   * 
   */
  void initialize_index_sets() override;

  /**
   * @brief This function only outputs some diagnostic data about the system matrix.
   * 
   */
  void validate();

  /**
   * @brief Reinitializes the data structures (solution vector, builds constraints, makes sparsity pattern, reinits the matrix).
   * 
   */
  auto reinit() -> void override;

  /**
   * @brief Reinits the right hand side vector.
   * 
   */
  auto reinit_rhs() -> void override;
  
  /**
   * @brief Computes the interface dofs index set for all the dofs on a surface of the inner domain.
   * 
   * @param interface_id 
   * @return dealii::IndexSet 
   */
  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  /**
   * @brief This level uses a direct solver (MUMPS) and this function computes the \f$LDL^T\f$ factorization it uses internally.
   * The solve function of LocalProblem objects are called sequentially in the sweeping preconditioner. The factorization only has to be computed once but that step is expensive. By providing this function we can call it in parallel on all LocalProblems resulting in perfect parallelization of the effort.
   * 
   */
  void compute_solver_factorization() override;

  /**
   * @brief Computes the L2 error of the solution that was computed last compaired to the exact solution of the problem.
   * 
   * Keep in mind that the "exact solution" for the waveguide case is a mode propagating on a straight waveguide, which is not applicable for a bent waveguide.
   * 
   * @return double Error value.
   */
  double compute_L2_error();

  /**
   * @brief Computes the L2 error and runs a time measurement around it.
   * 
   * @return double returns the error value.
   */
  double compute_error();

  /**
   * @brief All LocalProblem objects add up how often they have called their solver.
   * 
   * @return unsigned int Number of solver runs on the lowest level.
   */
  unsigned int compute_global_solve_counter() override;

  /**
   * @brief Frees up some memory from datastructures that are only required during the solution process to slim down the memory consumption after solving has terminated.
   * 
   */
  void empty_memory() override;

  /**
   * @brief Writes output of the solution of this problem including the boundary conditions and also provides a meta-file that can be used in Paraview to load all output by opening one file.
   * 
   * @param in_filename Name to use for the output file
   * @param transform If set to true, the output will be in the physical coordinate system.
   */
  void write_multifile_output(const std::string & in_filename, bool transform = false) override;

  void make_sparsity_pattern() override;
};
