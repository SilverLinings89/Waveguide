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

  dealii::IndexSet upper_interface_dofs;
  dealii::IndexSet lower_interface_dofs;
  dealii::SolverControl sc;
  unsigned int n_blocks_in_sweep;
  unsigned int index_in_sweep;
  unsigned int n_locally_active_dofs;
  dealii::IndexSet locally_active_dofs;
  KSP ksp;
  ComplexNumber * mpi_cache;
  bool is_mpi_cache_ready;
  PC pc;
  SampleShellPC shell;
  PetscInt* locally_owned_dofs_index_array;
  unsigned int lower_sweeping_interface_id;
  unsigned int upper_sweeping_interface_id;
  LocalMatrixPart local;
  unsigned int n_interface_dofs;
  std::vector<NumericVectorLocal> stored_solutions;
  
 public:
  NonLocalProblem(unsigned int);

  ~NonLocalProblem() override;

  auto reinit_mpi_cache(DofCount) -> void;

  auto initialize_own_dofs() -> void override;

  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  auto compute_lower_interface_id() -> BoundaryId;

  auto compute_upper_interface_id() -> BoundaryId;

  void assemble() override;

  void solve() override;

  void apply_sweep(Vec x_in, Vec x_out);

  void init_solver_and_preconditioner();

  void initialize() override;

  void initialize_index_sets() override;

  void reinit() override;

  NumericVectorLocal get_local_vector_from_global() override;
  
  auto get_center() -> Position const override;

  bool is_lowest_in_sweeping_direction();

  bool is_highest_in_sweeping_direction();

  void compute_solver_factorization() override;

  void reinit_rhs() override;
  
  DofOwner get_dof_owner(unsigned int id) override;

  auto print_dof_details(unsigned int dof) -> void;

  auto is_dof_locally_owned(unsigned int dof) -> bool;

  auto print_diagnosis_data() -> void;

  // New functions.

  auto reinit_u_vector(NumericVectorLocal * u) -> void;

  auto u_from_x_in(Vec x_in) -> NumericVectorLocal;

  auto S_inv(NumericVectorLocal u) -> NumericVectorLocal;

  auto lower_trace(NumericVectorLocal u) -> DofFieldTrace;

  auto upper_trace(NumericVectorLocal u) -> DofFieldTrace;

  auto send_down(DofFieldTrace trace_values) -> void;

  auto send_up(DofFieldTrace trace_values) -> void;

  auto receive_from_above() -> DofFieldTrace;

  auto receive_from_below() -> DofFieldTrace;

  auto vmult_down(const NumericVectorLocal u) -> NumericVectorLocal;

  auto vmult_up(const DofFieldTrace & u) -> NumericVectorLocal;

  auto trace_to_field(DofFieldTrace trace, BoundaryId b_id) -> NumericVectorLocal;

  auto subtract_fields(NumericVectorLocal a, NumericVectorLocal b) -> NumericVectorLocal;

  auto set_x_out_from_u(Vec x_out, NumericVectorLocal u_in) -> void;

  auto set_child_rhs_from_u(NumericVectorLocal u) -> void;

  auto set_u_from_child_solution(NumericVectorLocal * u)-> void;

  auto zero_upper_interface_dofs(NumericVectorLocal u)-> NumericVectorLocal;

  auto zero_lower_interface_dofs(NumericVectorLocal u)-> NumericVectorLocal;

  auto compute_interface_norm_for_u(NumericVectorLocal u, BoundaryId) -> double;

  std::string output_results();

  double compute_lower_interface_norm(NumericVectorLocal u);

  double compute_upper_interface_norm(NumericVectorLocal u);

  NumericVectorLocal sync_upwards(NumericVectorLocal u);

  NumericVectorLocal sync_downwards(NumericVectorLocal u);

  void store_solution(NumericVectorLocal u);

  void write_output_for_stored_solution(unsigned int index);

  void print_norm_distribution_for_vector(const NumericVectorDistributed & in_vector);

  NumericVectorLocal distribute_constraints_to_local_vector(const NumericVectorLocal u_in);

  void write_multifile_output(const std::string & filename, const NumericVectorDistributed field);

  void communicate_external_dsp(DynamicSparsityPattern * in_dsp);

  void make_sparsity_pattern() override;
};
