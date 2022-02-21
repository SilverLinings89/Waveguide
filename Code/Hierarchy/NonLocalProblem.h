#pragma once

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

class NonLocalProblem: public HierarchicalProblem {
private:
  dealii::SolverControl sc;
  unsigned int n_blocks_in_sweeping_direction;
  unsigned int index_in_sweeping_direction;
  unsigned int total_rank_in_sweep;
  unsigned int n_procs_in_sweep;
  NumericVectorDistributed dist_vector_1, dist_vector_2, dist_vector_3, u;

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
  bool is_shared_solution_up_to_date = false;
  
 public:
  NonLocalProblem(unsigned int);

  ~NonLocalProblem() override;

  void prepare_sweeping_data();

  void assemble() override;

  void solve() override;

  void apply_sweep(Vec x_in, Vec x_out);

  void init_solver_and_preconditioner();

  void initialize() override;

  void initialize_index_sets() override;

  void reinit() override;

  void compute_solver_factorization() override;

  void reinit_rhs() override;
  
  void S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst);

  auto set_x_out_from_u(Vec x_out) -> void;

  std::string output_results();

  void write_multifile_output(const std::string & filename, bool apply_coordinate_transform) override;

  void communicate_external_dsp(DynamicSparsityPattern * in_dsp);

  void make_sparsity_pattern() override;

  void set_u_from_vec_object(Vec in_v);

  void set_vector_from_child_solution(NumericVectorDistributed *);

  void set_child_rhs_from_vector(NumericVectorDistributed *);

  void print_vector_norm(NumericVectorDistributed * , std::string marker);

  void perform_downward_sweep();

  void perform_upward_sweep();

  void complex_pml_domain_matching(BoundaryId in_bid);

  void register_dof_copy_pair(DofNumber own_index, DofNumber child_index);

  ComplexNumber compute_signal_strength_of_solution();

  void update_shared_solution_vector();

  FEErrorStruct compute_global_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);

  void update_convergence_criterion(double last_residual) override;

  unsigned int compute_global_solve_counter() override;

  void reinit_all_vectors();

  unsigned int n_total_cells();

  double compute_h();

  unsigned int compute_total_number_of_dofs();

  std::vector<std::vector<ComplexNumber>> evaluate_solution_at(std::vector<Position>);

  void empty_memory() override;

  std::vector<double> compute_shape_gradient() override;
};
