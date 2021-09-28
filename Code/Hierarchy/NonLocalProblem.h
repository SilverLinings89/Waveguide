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
  unsigned int n_blocks_in_sweep;
  unsigned int index_in_sweep;
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
  
 public:
  NonLocalProblem(unsigned int);

  ~NonLocalProblem() override;

  void assemble() override;

  void solve() override;

  void apply_sweep(Vec x_in, Vec x_out);

  void init_solver_and_preconditioner();

  void initialize() override;

  void initialize_index_sets() override;

  void reinit() override;

  void compute_solver_factorization() override;

  void reinit_rhs() override;
  
  // New functions.

  void S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst, bool execute_locally);

  auto set_x_out_from_u(Vec x_out, NumericVectorDistributed * u_in) -> void;

  auto set_u_from_child_solution(NumericVectorDistributed * u)-> void;

  std::string output_results();

  void store_solution(NumericVectorLocal u);

  void write_output_for_stored_solution(unsigned int index);

  void write_multifile_output(const std::string & filename, const NumericVectorDistributed field);

  void communicate_external_dsp(DynamicSparsityPattern * in_dsp);

  void make_sparsity_pattern() override;

  NumericVectorDistributed vector_from_vec_obj(Vec in_v);

  void set_vector_from_child_solution(NumericVectorDistributed *);

  void set_child_rhs_from_vector(NumericVectorDistributed *);

  void copy_local_part(NumericVectorDistributed * src, NumericVectorDistributed * dst);

  void subtract_vectors(NumericVectorDistributed * a, NumericVectorDistributed * b);

  NumericVectorDistributed off_diagonal_product(unsigned int i, unsigned int j, NumericVectorDistributed *);

  void print_vector_norm(NumericVectorDistributed * , std::string marker);
};
