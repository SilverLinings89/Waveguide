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
#include "../Helpers/Enums.h"

class NonLocalProblem: public HierarchicalProblem {
private:
  std::array<bool, 6> is_sweeping_hsie_surface;
  DofCount total_number_of_dofs_on_level;
  dealii::IndexSet upper_interface_dofs;
  dealii::IndexSet lower_interface_dofs;
  dealii::SolverControl sc;
  KSP ksp;
  ComplexNumber * u;
  std::array<std::vector<std::pair<DofNumber, DofNumber>>,6> coupling_dofs;
  ComplexNumber * mpi_cache;
  bool is_mpi_cache_ready;
  PC pc;
  SampleShellPC shell;
  PetscInt* locally_owned_dofs_index_array;
  unsigned int lower_sweeping_interface_id;
  unsigned int upper_sweeping_interface_id;
  
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

  void solve() override;

  void apply_sweep(Vec x_in, Vec x_out);

  void init_solver_and_preconditioner();

  void initialize() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  void reinit() override;

  NumericVectorLocal get_local_vector_from_global() override;

  auto get_center() -> Position const override;

  void H_inverse();

  NumericVectorLocal extract_local_upper_dofs();

  NumericVectorLocal extract_local_lower_dofs();

  void send_local_lower_dofs(std::vector<ComplexNumber>);

  void receive_local_lower_dofs_and_H();

  void send_local_upper_dofs(std::vector<ComplexNumber>);

  void receive_local_upper_dofs();

  bool is_lowest_in_sweeping_direction();

  bool is_highest_in_sweeping_direction();

  void propagate_up();

  void compute_solver_factorization() override;
  
  void output_results() override;

  void update_mismatch_vector(BoundaryId) override;

  auto make_sparsity_pattern_for_surface(unsigned int, DynamicSparsityPattern *) -> void;

  void fill_dsp_over_mpi(BoundaryId surface, dealii::DynamicSparsityPattern * in_dsp);

  auto UpperBlockProductAfterH() -> std::vector<ComplexNumber>;

  auto LowerBlockProduct() -> std::vector<ComplexNumber>;

  void setSolutionFromVector(Vec x_in);

  void setChildSolutionComponentsFromU();

  void setChildRhsComponentsFromU();

  void reinit_rhs() override;
  
  DofOwner get_dof_owner(unsigned int id) override;

  auto print_dof_details(unsigned int dof) -> void;

  auto is_dof_locally_owned(unsigned int dof) -> bool;

  auto print_diagnosis_data() -> void;

};
