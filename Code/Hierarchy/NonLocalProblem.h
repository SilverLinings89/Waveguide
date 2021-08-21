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
  KSP ksp;
  ComplexNumber * mpi_cache;
  bool is_mpi_cache_ready;
  PC pc;
  SampleShellPC shell;
  PetscInt* locally_owned_dofs_index_array;
  unsigned int lower_sweeping_interface_id;
  unsigned int upper_sweeping_interface_id;
  LocalMatrixPart local;
  
 public:
  NonLocalProblem(unsigned int);

  ~NonLocalProblem() override;

  auto reinit_mpi_cache(DofCount) -> void;

  auto initialize_own_dofs() -> void override;

  auto compute_lower_interface_dof_count() -> DofCount override;

  auto compute_upper_interface_dof_count() -> DofCount override;

  DofCount compute_interface_dofs(BoundaryId interface_id);

  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  auto compute_lower_interface_id() -> BoundaryId;

  auto compute_upper_interface_id() -> BoundaryId;

  void assemble() override;

  void assemble_local_system();

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

  auto vmult_down(NumericVectorLocal u) -> NumericVectorLocal;
  auto vmult_up(DofFieldTrace u) -> NumericVectorLocal;

  auto trace_to_field(DofFieldTrace trace, BoundaryId b_id) -> NumericVectorLocal;

  auto subtract_fields(NumericVectorLocal a, NumericVectorLocal b) -> NumericVectorLocal;

  auto set_x_out_from_u(Vec x_out, NumericVectorLocal u_in) -> void;

  auto set_child_solution_from_u(NumericVectorLocal u) -> void;

  auto set_child_rhs_from_u(NumericVectorLocal u, bool add_onto_child_rhs) -> void;

  auto set_u_from_child_solution(NumericVectorLocal * u)-> void;

  auto zero_upper_interface_dofs(NumericVectorLocal u)-> NumericVectorLocal;

  auto zero_lower_interface_dofs(NumericVectorLocal u)-> NumericVectorLocal;

  void update_u_from_trace(NumericVectorLocal * in_u, DofFieldTrace trace, bool from_lower);

  void receive_local_lower_dofs_and_H();

  auto compute_interface_norm_for_u(NumericVectorLocal u, BoundaryId) -> double;

  std::string output_results();
};
