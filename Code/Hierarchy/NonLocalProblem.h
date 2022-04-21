#pragma once

/**
 * @file NonLocalProblem.h
 * @author Pascal Kraft
 * @brief This file includes the class NonLocalProblem which is the essential class for the hierarchical sweeping preconditioner.
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Core/Types.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <mpi.h>
#include <complex>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Core/Enums.h"

/**
 * @brief The NonLocalProblem class is part of the sweeping preconditioner hierarchy. 
 * 
 * It assembles a system-matrix and right-hand side and solves it using a GMRES solver. It also handles all the communication required to perform that task and assembles sparsity patterns.
 * 
 */

class NonLocalProblem: public HierarchicalProblem {
private:
  dealii::SolverControl sc;
  unsigned int n_blocks_in_sweeping_direction;
  unsigned int index_in_sweeping_direction;
  unsigned int total_rank_in_sweep;
  unsigned int n_procs_in_sweep;
  NumericVectorDistributed dist_vector_1, dist_vector_2, dist_vector_3, u;
  NumericVectorDistributed adjoint_state;
  bool has_adjoint = false;

  dealii::IndexSet locally_active_dofs;
  KSP ksp;
  PC pc;
  SampleShellPC shell;
  PetscInt* locally_owned_dofs_index_array;
  std::vector<NumericVectorLocal> stored_solutions;
  unsigned int n_locally_active_dofs;
  unsigned int step_counter = 0;
  std::vector<unsigned int> vector_copy_own_indices;
  std::vector<unsigned int> vector_copy_child_indeces;
  std::vector<ComplexNumber> vector_copy_array;
  double internal_vector_norm = 0.0;
  dealii::LinearAlgebra::distributed::Vector<ComplexNumber> shared_solution;
  dealii::LinearAlgebra::distributed::Vector<ComplexNumber> shared_adjoint;
  bool is_shared_solution_up_to_date = false;
  
 public:

  /**
   * @brief Construct a new Non Local Problem object using a level value as input.
   * The constructor of this class actually performs several tasks: Initialize the solver control (which performs convergence tests), start initialization of the remaining objects of the sweeping hierarchy (NonlocalProblem(3) calls NonlocalProblem(2) calls NonLocalProblem(1) calls LocalProeblm()). Additionally, it determines the correct sweeping direction and initializes cached values of neighbors and the matrix.
   * Next it prepares the locally active dof set and builds an output object for residuals of the own GMRES solver.
   * @param level 
   */
  NonLocalProblem(unsigned int level);

  /**
   * @brief Destroy the Non Local Problem object
   * This means deleting the matrix and locally owned dofs index array as well as the KSP object in PETSC.
   * 
   */
  ~NonLocalProblem() override;

  /**
   * @brief Computes some basic information about the sweep like the number of processes in the sweeping direction as well as the own index in that direction.
   * 
   */
  void prepare_sweeping_data();

  /**
   * @brief Calls assemble on the InnerProblem and the boundary methods.
   * Steps: First reset the system matrix and rhs to zero (for the optimization cases). Then start a timer. Call assemble_systeem on the InnerDomain and fill_matrix on the boundary contributions. Then stop the timer. Finally compress the datastructures and update the PETSC ksp object to recognize the new operator.
   */
  void assemble() override;

  /**
   * @brief Solves using a GMRES solver with a sweeping preconditioner. 
   * The Sweeping preconditioner is also implemented in this class and calls on the child object for the next level.
   * The included direct solver call can only occur if it is hard-coded to do so or the parameter use_direct_solver was set. This is only intended for debugging use.
   * The function also uses a timer and generates output on the main stream of the application.
   */
  void solve() override;

  /**
   * @brief Similar to solve() but uses the adjoint solution for the output of the solution.
   * 
   */
  void solve_adjoint() override;

  /**
   * @brief Cor function of the sweeping preconditioner. Applies the preconditioner to an input vector and returns the result in the second argument
   * 
   * This function has been refactored to be easier to read. This formulation is in line with the algorithm formulations in the dissertation documents.
   * 
   * @param x_in The vector the preconditioner should be applied to.
   * @param x_out The vector storing the result.
   */
  void apply_sweep(Vec x_in, Vec x_out);

  /**
   * @brief Prepares the PETSC objects required for the computation.
   * 
   * This code relies on PETSC to perform the computationally expensive tasks. We use itterative solvers from this library.
   * This function sets up the Krylov Space wrapper for the solvers (KSP) which is default for PETSC applications and also provides the preconditioner to the object. The NonLocalProblem object contains all required functions for the evalutaion of the preconditioner and the constructed preconditioner object (PC) simply references those (In detail: A Batch-Preconditioner is initialized which is a way of wrapping a function call and providing it as a preconditioner).
   * Additionally, it sets the operator used in the solver to the system matrix constructed for the NonLocalProblem. In the next step it provides the individual solver with necessary data depending on its type. For example: For GMRES we set the restart parameter and the preconditioner side. 
   * 
   */
  void init_solver_and_preconditioner();

  /**
   * @brief Recursive. Prepares all datastructures.
   * 
   * At the point of this function call, the NonLocalProblem object can access the dof distribution on the current level and we can therefore prepare vectors and matrices as well as sparsity patterns. The function also calls itself on the child level.
   * 
   */
  void initialize() override;

  /**
   * @brief Part of the initialization hierarchy.
   * 
   * Sets the locally cached values of the owned dofs and prepares a petsc index array for efficient extraction of dof values from vectors.
   * 
   */
  void initialize_index_sets() override;

  /**
   * @brief Builds constraints and sparsity pattern, then initializes the matrix and some cached data for faster data access.
   * Matrix initialization is a complex step for large runs because large memory consumtion is expected.
   * 
   */
  void reinit() override;

  /**
   * @brief Recursive. This function only propagates to the child. On the lowest level (which is a LocalProblem), this will prepare the direct solver factorization.
   * 
   */
  void compute_solver_factorization() override;

  /**
   * @brief Prepare the data structure which stores the right hand side vector.
   * 
   */
  void reinit_rhs() override;
  
  /**
   * @brief Applies the operator \f$S^{-1}\f$ to the provided src vector and returns the result in dst.
   * This is the function call in the preconditioner that calls the solver of the child problem.
   * 
   * @param src The vector the child solver should be applied to.
   * @param dst The vector to store the result in.
   */
  void S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst);

  /**
   * @brief Set the x out from u object
   * We use different data types for computation in our own code then the somewhat clunky PETSC data types. Therefore, once we are done computing the output vector of the sweeping preconditioner application to an input vector in our own data-type, we have to update the provided output vector, which is a PETSC data structure.
   * This function performs no math only copying of the vector to the appropirate output format.
   * @param x_out 
   */
  auto set_x_out_from_u(Vec x_out) -> void;

  /**
   * @brief Writes output files about the run on this level.
   * This calls another function which performs the actual writing of the output. This function mainly generates a vector of all locally active dofs (they might be stored on another process) and makes it available locally. It also logs signal strength and solver data.
   * @return std::string empty string in this case.
   */
  std::string output_results();

  /**
   * @brief Generates actual output files about the current levels solution.
   * For a given filename this function writes the vtu and vtk output files for the inner domain and the boundary methods (if they are PML). It keeps track of all the generated files and generates a header file for Paraview which loads all the individual files. If the input flaf transformed is true, it does the same for the solution in the physical coordinate sysytem.
   * 
   * @param filename Base part of the output file names.
   * @param apply_coordinate_transform if true, the output will be in transformed coordinates.
   */
  void write_multifile_output(const std::string & filename, bool apply_coordinate_transform) override;

  /**
   * @brief Exchange non-zero entries of the system matrix across neighboring processes.
   * This is an important function and reasonably complex. However, it mainly handles the exchange of data in the sparsity pattern and is not mathematical in nature.
   * @param in_dsp The dsp to fill.
   */
  void communicate_external_dsp(DynamicSparsityPattern * in_dsp);

  /**
   * @brief Determines the non-zero entries of the system matrix and prepares a sparsity pattern object that stores this information for efficient memory allocation of the matrices.
   * 
   */
  void make_sparsity_pattern() override;

  /**
   * @brief Turns the input PETSC vector, the sweeping preconditioner should be applied to into a data structure that works well in deal.II.
   * 
   * @param in_v The vector.
   */
  void set_u_from_vec_object(Vec in_v);

  /**
   * @brief Copies the solution of a child solver run up one hierarchy level.
   * 
   * @param vec The vector to store the child solution in on this level.
   */
  void set_vector_from_child_solution(NumericVectorDistributed * vec);

  /**
   * @brief Copies a rhs vector down to the child vector befor calling solve on it.
   * 
   */
  void set_child_rhs_from_vector(NumericVectorDistributed *);

  /**
   * @brief Outputs the L2 norm of a provided vector.
   * 
   * @param vec The vector to measure
   * @param marker A string marker that will be part of the output so it can be identified in the logs.
   */
  void print_vector_norm(NumericVectorDistributed * vec, std::string marker);

  /**
   * @brief Performs the first half of the sweeping preconditioner.
   * The code looks more bloated than in the pseudo-code algorithm but most of it is just vector storage management.
   */
  void perform_downward_sweep();

  /**
   * @brief Performs the second half of the sweeping preconditioner.
   * The code looks more bloated than in the pseudo-code algorithm but most of it is just vector storage management.
   */
  void perform_upward_sweep();

  /**
   * @brief PML domains are sometimes different across the hierarchy. Whenever we copy a vector up or down we have to match the indices correctly.
   * 
   * This function prepares index pairs across the hierarchy that reference the same dof on different levels. It only performs this task for one boundary and builds the mapping for dofs on the current level and the immediate child.
   * 
   * The data is stored in the vector_copy_own_indices, vector_copy_child_indices and vector_copy_array. These datastructures are always used when we call functions like set_child_rhs_from_vector.
   * 
   * @param in_bid The surface to perform this task on.
   */
  void complex_pml_domain_matching(BoundaryId in_bid);

  /**
   * @brief Used by complex_pml_domain_matching to register a degree of freedom that has the index own_index on this level and child_index in the child.
   * 
   * Whenever a vector is copied between the child and this, the dof child_index on the child and own_index on this will have the same value.
   * 
   * @param own_index Index on this.
   * @param child_index Index on the child.
   */
  void register_dof_copy_pair(DofNumber own_index, DofNumber child_index);

  /**
   * @brief Computes how strong the signal is on the output connector.
   * 
   * @return ComplexNumber Phase and amplitude of the signal.
   */
  ComplexNumber compute_signal_strength_of_solution();

  /**
   * @brief Not all locally active dofs (dofs that couple to locally owned ones) are locally owned. For output operations we need to access all these values from local memory. This function gathers all non-locally-owned dof values and stores them in a purely local vector.
   * 
   */
  void update_shared_solution_vector();

  /**
   * @brief Computes the L2 error of the provided vector solution agains a theoretical solution of the current problem.
   * 
   * @param in_solution The solution vector.
   * @return FEErrorStruct A structure containing the L2 error.
   */
  FEErrorStruct compute_global_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);

  /**
   * @brief To be able to abort early on child solvers, we need to store the current residual on the current level.
   * This value can then be accessed by a child solver to determine its abort condition.
   * 
   * @param last_residual Latest computed local residual.
   */
  void update_convergence_criterion(double last_residual) override;

  /**
   * @brief Adds up the number of solver calls on the current level
   * 
   * @return unsigned int How often the solver was called on this level.
   */
  unsigned int compute_global_solve_counter() override;

  /**
   * @brief Reinits all vectors on the current vector.
   * 
   */
  void reinit_all_vectors();

  /**
   * @brief Computes the number of cells of the local part of the current problem and then adds these valus for all processes in the current sweep.
   * 
   * @return unsigned int Number of cells on this level.
   */
  unsigned int n_total_cells();

  /**
   * @brief Computes the mesh constant of the local level problem.
   * 
   * @return double Mesh size constant for the triangulation.
   */
  double compute_h();

  /**
   * @brief Computes the total number of dofs on the current level (not only the locally owned part).
   * 
   * @return unsigned int Number of dofs on this level.
   */
  unsigned int compute_total_number_of_dofs();

  /**
   * @brief Computes the E-field evaluation at all the positions in the input vector and returns a vector of the same length with the values.
   * 
   * @param locations A vector containing a set of positions that must be part of the local triangulation.
   * @return std::vector<std::vector<ComplexNumber>> Vector of e-field evaluations for the provided locations.
   */
  std::vector<std::vector<ComplexNumber>> evaluate_solution_at(std::vector<Position> locations);

  /**
   * @brief Reduces the memory consumption of local data structures to save memory once compuataions are done. This deletes, among other things, the factorization in direct solvers.
   * 
   */
  void empty_memory() override;

  /**
   * @brief Computes the shape gradient contributions of this process. 
   * 
   * The i-th entry in this vector is the derivative of the loss functional by the i-th degree of freedom of the shape.
   * 
   * @return std::vector<double> 
   */
  std::vector<double> compute_shape_gradient() override;

  /**
   * @brief Set the rhs for the computation of the adjoint state.
   * 
   */
  void set_rhs_for_adjoint_problem();
};
