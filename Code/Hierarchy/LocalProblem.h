#pragma once

/**
 * @file LocalProblem.h
 * @author Pascal Kraft
 * @brief Contains the class LocalProblem, the lowest level of the sweeping hierarchy.
 */
#include <string>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "HierarchicalProblem.h"
#include "../Core/InnerDomain.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "deal.II/lac/petsc_solver.h"
#include <array>
#include <memory>

/**
 * @brief The problem on level 0 of the hierarchy: one subdomain on one process, solved with a direct solver.
 *
 * The subdomain consists of the inner domain of the process and the boundary methods on its surfaces. The matrix
 * is factorized once (by default with MUMPS) and the factorization is reused for all solves that the sweeping
 * preconditioners of the higher levels request.
 */
class LocalProblem: public HierarchicalProblem {
public:
  /**
   * The direct solver of the subdomain. It is a PETSc KSP of type "preonly" whose preconditioner is a complete
   * factorization. The defaults (an LDL^T factorization of the complex symmetric matrix computed by MUMPS) can be
   * changed at run time with PETSc options that carry the prefix "local_", for example
   *   -local_pc_type lu -local_pc_factor_mat_solver_type superlu_dist
   *   -local_mat_mumps_icntl_35 2 -local_mat_mumps_cntl_7 1e-6     (block low rank compression in MUMPS)
   * This is also the place to plug in a GPU solver (see MODERNIZATION.md).
   */
  KSP local_ksp = nullptr;

  /**
   * @brief L2 norm of the exact solution (the source field) on the interior domain. Used to normalize the error.
   */
  double compute_L2_norm_of_exact_solution();

  /**
   * Statistics of the direct solver of this subdomain. They are collected in solve() and written to results.txt.
   */
  double time_of_factorization = 0;      // wall time of the calls that computed a factorization (includes one solve)
  /** @brief Accumulated wall time in seconds of all solves that reused an existing factorization. */
  double time_of_solves = 0;             // accumulated wall time of all other calls
  /** @brief Number of solves that contributed to time_of_solves. */
  unsigned int n_timed_solves = 0;
  /** @brief True if a factorization of the current matrix exists. Reset by assemble(), set by the first solve after it. */
  bool has_factorization = false;
  /** @brief Number of dof indices of level 0 that belong to no element. assemble() gives them the equation u_i = 0. They are subtracted when the number of unknowns is reported. */
  unsigned int n_unused_dof_indices = 0;  // dof indices without any equation (see assemble())
  /** @brief Number of entries of the factors without compression, MUMPS INFOG(29). 0 if the solver is not MUMPS. */
  double factor_entries_full_rank = 0;   // MUMPS INFOG(29): entries of the factors without compression
  /** @brief Number of entries of the factors that are stored, MUMPS INFOG(35). Equal to factor_entries_full_rank without block low rank compression. */
  double factor_entries_stored = 0;      // MUMPS INFOG(35): entries that are actually stored (equal without BLR)
  /** @brief Memory used during the factorization in megabytes, MUMPS INFOG(22). */
  double factor_memory_mb = 0;           // MUMPS INFOG(22): memory used during the factorization in megabytes
  
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

  /** @brief Builds the sparsity pattern of the local system from the inner domain and the boundary methods. */
  void make_sparsity_pattern() override;
};
